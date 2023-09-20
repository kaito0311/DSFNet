import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import config
from src.dataset.dataloader import make_dataset, img_to_tensor
from src.model.loss import *
from src.util.printer import DecayVarPrinter
from tqdm import tqdm
import skimage.io as io
from src.faceutil import mesh
import matplotlib.pyplot as plt 
from src.visualize.render_mesh import render_face_orthographic, render_uvm
from src.visualize.plot_verts import plot_kpt, compare_kpt

class BasePredictor:
    def __init__(self, weight_path):
        self.model = self.get_model(weight_path)

    def get_model(self, weight_path):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError


class DSFNetPredictor(BasePredictor):
    def __init__(self, weight_path):
        super(DSFNetPredictor, self).__init__(weight_path)

    def get_model(self, weight_path):
        from src.model.DSFNet import get_model
        model = get_model()
        pretrained = torch.load(weight_path, map_location=config.DEVICE)
        model.load_state_dict(pretrained)
        model = model.to(config.DEVICE)
        model.eval()
        return model


class Evaluator:
    def __init__(self):
        self.all_eval_data = None
        self.metrics = {"nme3d": NME(),
                        "nme2d": NME2D(),
                        "kpt2d": KptNME2D(),
                        "kpt3d": KptNME(),
                        "rec": RecLoss(),
                        }
        self.printer = DecayVarPrinter()

    def get_data(self):
        val_dataset = make_dataset(config.VAL_DIR, 'val')
        self.all_eval_data = val_dataset.val_data
    
    def show_face_uvm(self, face_uvm, img, gt_uvm=None, is_show=True):
        ret = render_uvm(face_uvm, img)
        if is_show:
            plt.imshow(ret)
            plt.show()
        ret_kpt = plot_kpt(img, face_uvm[uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]])
        if is_show:
            plt.imshow(ret_kpt)
            plt.show()
        if gt_uvm is not None:
            ret_cmp = compare_kpt(face_uvm, gt_uvm, img)
            if is_show:
                plt.imshow(ret_cmp)
                plt.show()
            return ret, ret_kpt, ret_cmp
        else:
            return ret, ret_kpt
    def evaluate(self, predictor):
        with torch.no_grad():
            predictor.model.eval()
            self.printer.clear()

            pred_angles = np.zeros((len(self.all_eval_data),3))
            valid_idx = np.arange(len(self.all_eval_data))
            for i in tqdm(valid_idx):
                item = self.all_eval_data[i]
                init_img = item.get_image()
                image = (init_img / 255.0).astype(np.float32)
                for ii in range(3):
                    image[:, :, ii] = (image[:, :, ii] - image[:, :, ii].mean()) / np.sqrt(
                        image[:, :, ii].var() + 0.001)
                image = img_to_tensor(image).to(config.DEVICE).float().unsqueeze(0)
        
                init_pos_map = (item.get_pos_map())
                pos_map = init_pos_map / config.POSMAP_FIX_RATE
                pos_map = img_to_tensor(pos_map).float().to(config.DEVICE).unsqueeze(0)

                out = predictor.model(inpt={'img': image}, targets={}, mode='predict')
                pred_a = mesh.transform.matrix2angle(out['R_rot'].cpu().detach().numpy()[0].T)
                pred_a = np.array([pred_a[0],pred_a[1],pred_a[2]])
                pred_a[0] = -pred_a[0]
                pred_angles[i] = pred_a
                print("out face_uvm : ", out['face_uvm']) 

                output_folder = "./src/output/"
                face_uvm_out = out['face_uvm'][0].cpu().permute(1, 2, 0).numpy() * config.POSMAP_FIX_RATE
                ret, ret_kpt = self.show_face_uvm(face_uvm_out, init_img, None, True)
                # io.imsave(f'{output_folder}/{i}_cmp.jpg', ret_cmp)
                io.imsave(f'{output_folder}/{i}_kpt.jpg', ret_kpt)
                io.imsave(f'{output_folder}/{i}_face.jpg', ret)
                io.imsave(f'{output_folder}/{i}_img.jpg', init_img) 


                for key in self.metrics:
                    func = self.metrics[key]
                    error = func(pos_map, out['face_uvm']).cpu().numpy()
                    self.printer.update_variable_avg(key, error)


        print('Dataset Results')
        return_dict = {}
        for key in self.metrics:
            print(self.printer.get_variable_str(key))
            return_dict[key] = float(self.printer.get_variable_str(key).split(' ')[1])
            
        # head_pose_estimation = benchmark_FOE(pred_angles,valid_idx)
        # for key in head_pose_estimation:
        #     return_dict[key] = round(head_pose_estimation[key],3)
    
        return return_dict
    

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.get_data()
    predictor = DSFNetPredictor(config.PRETAINED_MODEL)
    evaluator.evaluate(predictor)
