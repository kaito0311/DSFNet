import scipy.io as sio
import numpy as np 
import os

# model_bfm_result = sio.loadmat("data/Out/BFM.mat")['model'][0,0]
# model_infor_result = sio.loadmat('data/Out/BFM_info.mat')['model_info'][0,0]
# uv_result = sio.loadmat("data/Out/BFM_UV.mat")['UV']


model_bfm_mat = sio.loadmat('raw/01_MorphableModel.mat')
model_infor_mat = sio.loadmat('3ddfa/model_info.mat')
expression_mat = sio.loadmat("3ddfa/Model_Expression.mat")
tri_mouth_mat = sio.loadmat("3ddfa/Model_tri_mouth.mat")
face_contour_mat = sio.loadmat("3ddfa/Model_face_contour_trimed.mat")
nose_hold_mat = sio.loadmat("3ddfa/Modelplus_nose_hole.mat")
keypoint_parallel = sio.loadmat("3ddfa/Modelplus_parallel.mat")
bfm_uv_mat = sio.loadmat("stn/BFM_UV.mat")

print("bfm model keys: ", model_bfm_mat.keys())
print("model infor keys: ", model_infor_mat.keys())

trimIndex = model_infor_mat['trimIndex']

trimIndex = np.array(trimIndex, dtype=np.int64)
trimIndex_f = np.concatenate([3 * trimIndex - 2, 3 * trimIndex - 1, 3 * trimIndex], axis=1).T -1 
trimIndex_f = trimIndex_f.flatten(order='F')

list_for_model = {}
list_for_model_infor = {} 

#TM-ANCHOR: create model 
list_for_model["shapeMU"] = model_bfm_mat['shapeMU'][trimIndex_f, :]
list_for_model['shapePC'] = model_bfm_mat['shapePC'][trimIndex_f, :]
list_for_model['shapeEV'] = model_bfm_mat['shapeEV']
list_for_model['texMU'] = model_bfm_mat['texMU'][trimIndex_f, :] 
list_for_model['texPC'] = model_bfm_mat['texPC'][trimIndex_f, :]
list_for_model['texEV'] = model_bfm_mat['texEV']
list_for_model['tri'] = model_infor_mat['tri']
list_for_model["kpt_ind"] = model_infor_mat['keypoints']





#TM-ANCHOR: create model infor

list_for_model_infor["kpt_ind"] = model_infor_mat['keypoints']
list_for_model_infor["trimIndex"] = model_infor_mat['trimIndex']
list_for_model_infor["symlist"] = model_infor_mat['symlist']
list_for_model_infor["symlist_tri"] = model_infor_mat['symlist_tri']
list_for_model_infor["segbin"] = model_bfm_mat['segbin'][trimIndex.flatten() - 1, :].T # start index in matlab is 1 not 0 
list_for_model_infor["segbin_tri"] = model_infor_mat['segbin_tri'].T
        

# TM-ANCHOR: load expression 
list_for_model['expMU'] =  expression_mat['mu_exp']
list_for_model['expPC'] = expression_mat['w_exp']
list_for_model["expEV"] = expression_mat['sigma_exp']

# TM-ANCHOR: load mouth tri 

# print(tri_mouth_mat['tri_mouth'].shape)
list_for_model["tri_mouth"] = tri_mouth_mat['tri_mouth']
list_for_model_infor["tri_mouth"] = tri_mouth_mat['tri_mouth']

# TM-ANCHOR: face contour 
list_for_model_infor["face_contour"] = face_contour_mat['face_contour']
list_for_model_infor["face_contour_line"] = face_contour_mat['face_contour_line']
list_for_model_infor["face_contour_front"] = face_contour_mat['face_contour_front']
list_for_model_infor["face_contour_front_line"] = face_contour_mat['face_contour_front_line']

# TM-ANCHOR: nose hole
list_for_model_infor["nose_hole"] = nose_hold_mat['nose_hole']
list_for_model_infor["nose_hole_right"] = nose_hold_mat['nose_hole_right']
list_for_model_infor["nose_hole_left"] = nose_hold_mat['nose_hole_left']

# TM-ANCHOR: parallel for key points 
list_for_model_infor["parallel"] = keypoint_parallel['parallel']
list_for_model_infor["parallel_face_contour"] = keypoint_parallel['parallel_face_contour']

# TM-ANCHOR: load 3DMMasSTN UV coords 
uv_coords = bfm_uv_mat['UV'][trimIndex.flatten() - 1, :].T

# modify bad vers 
bad_ind = np.array([10032, 10155, 10280], dtype= np.int64)
bad_ind = bad_ind - 1 
round1 = np.array([10033, 10158 ], dtype= np.int64)
round2 = np.array([10534, 10157, 10661], dtype= np.int64)
round3 = np.array([10916, 10286], dtype= np.int64)

uv_coords[:, bad_ind[0]] = np.mean(uv_coords[:, round1 - 1], 1) # start index of matlab is 1 not 0 as python
uv_coords[:, bad_ind[1]] = np.mean(uv_coords[:, round2 - 1], 1)
uv_coords[:, bad_ind[2]] = np.mean(uv_coords[:, round3 - 1], 1)

list_for_model_infor["uv_coords"] = uv_coords.T

# modify tri mouth 
tm_inner = tri_mouth_mat['tri_mouth']
tm_inner_add = np.array(
    [
        [6420, 6542, 6664],
        [6420, 6294, 6167],
        [6167, 6297, 6420],
        [6167, 6297, 6296],
        [6167, 6296, 6295],
        [6167, 6295, 6039],
        [6168, 6295 ,6039]
    ]
)

ind_bad = 38 - 1

all_ind = range(0, tm_inner.shape[1])

tm_inner = tm_inner[:, np.setdiff1d(all_ind, bad_ind)]
tm_inner = np.concatenate([tm_inner, tm_inner_add.T], axis=1)

list_for_model["tri_mouth"] = tm_inner 
list_for_model_infor["tri_mouth"] = tm_inner 




# for index in range(len(list_for_model)):
#     myself_item = list_for_model[index] 
#     target_item = model_bfm_result[index]
#     # print(index)
#     assert np.sum(myself_item - target_item) == 0 

# for index in list(range(14))+ [16]:
#     myself_item = list_for_model_infor[index] 
#     target_item = model_infor_result[index]
#     assert np.sum(myself_item - target_item) == 0, "haiiz" + {index}

# index = 14 
# assert np.sum([np.sum(np.sum(list_for_model_infor[index][i] - model_infor_result[index][i])) for i in range(17)]) == 0 
# index = 15 
# assert np.sum([np.sum(np.sum(list_for_model_infor[index][i] - model_infor_result[index][i])) for i in range(17)]) == 0 

# assert len(list_for_model) == len(model_bfm_result)
# assert len(list_for_model_infor) == len(model_infor_result)

# assert np.sum(uv_result - uv_coords.T) == 0 


# save
BFM_model = {} 
BFM_model['model'] = list_for_model

sio.savemat(
    "Out/BFM.mat",
    BFM_model    
)

BFM_infor = {}
BFM_infor['model_info'] = list_for_model_infor

sio.savemat(
    "Out/BFM_info.mat",
    BFM_infor
)

BFM_UV = {} 
BFM_UV["UV"] = uv_coords.T

sio.savemat(
    "Out/BFM_UV.mat",
    BFM_UV
)


cmd = "cp 3ddfa/vertex_code.mat Out/pncc_code.mat"
os.system(cmd)

