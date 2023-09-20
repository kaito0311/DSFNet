import numpy as np
import scipy.io as sio
import shutil

# addpath(genpath(pwd))
# --> model

# load raw BFM models
bfm_model = sio.loadmat('raw/01_MorphableModel.mat')
print(bfm_model.keys())
# load 3ddfa data
# 1. load bfm information. trim
model_info = sio.loadmat('3ddfa/model_info.mat')
print(model_info.keys())
model_info['trimIndex'] -= 1 
trimIndex_f = np.concatenate((3 * model_info['trimIndex'] - 2, 3 * model_info['trimIndex'] - 1, 3 * model_info['trimIndex']), axis=1).T
trimIndex_f = trimIndex_f.flatten()

model = {}
model['shapeMU'] = bfm_model['shapeMU'][trimIndex_f, :]
model['shapePC'] = bfm_model['shapePC'][trimIndex_f, :]
model['shapeEV'] = bfm_model['shapeEV']
model['texMU'] = bfm_model['texMU'][trimIndex_f, :]
model['texPC'] = bfm_model['texPC'][trimIndex_f, :]
model['texEV'] = bfm_model['texEV']
model['tri'] = model_info['tri']
model['kpt_ind'] = model_info['keypoints']

model_info['kpt_ind'] = model_info['keypoints']
model_info['trimIndex'] = model_info['trimIndex']
model_info['symlist'] = model_info['symlist']
model_info['symlist_tri'] = model_info['symlist_tri']
# segbin: nose eyes mouth rest
model_info['segbin'] = model_info['segbin'][model_info['trimIndex'], :].T
model_info['segbin_tri'] = model_info['segbin_tri'].T

# 2. load expression
expression_model = sio.loadmat('3ddfa/Model_Expression.mat')
model['expMU'] = expression_model['mu_exp']
model['expPC'] = expression_model['w_exp']
model['expEV'] = expression_model['sigma_exp']

# 3. load mouth tri
mouth_tri = sio.loadmat('3ddfa/Model_tri_mouth.mat')
model['tri_mouth'] = mouth_tri['tri_mouth']
model_info['tri_mouth'] = mouth_tri['tri_mouth']

# 4. face contour
face_contour = sio.loadmat('3ddfa/Model_face_contour_trimed.mat')
model_info['face_contour'] = face_contour['face_contour']
model_info['face_contour_line'] = face_contour['face_contour_line']
model_info['face_contour_front'] = face_contour['face_contour_front']
model_info['face_contour_front_line'] = face_contour['face_contour_front_line']

# 5. nose hole
nose_hole = sio.loadmat('3ddfa/Modelplus_nose_hole.mat')
model_info['nose_hole'] = nose_hole['nose_hole']
model_info['nose_hole_right'] = nose_hole['nose_hole_right']
model_info['nose_hole_left'] = nose_hole['nose_hole_left']

# 6. parallel for key points
parallel = sio.loadmat('3ddfa/Modelplus_parallel.mat')
model_info['parallel'] = parallel['parallel']
model_info['parallel_face_contour'] = parallel['parallel_face_contour']

# 7. pncc
shutil.copyfile('3ddfa/vertex_code.mat', 'Out/pncc_code.mat')

# load 3DMMasSTN UV coords
bfm_uv = sio.loadmat('stn/BFM_UV.mat')
uv_coords = bfm_uv['UV'][model_info['trimIndex'][:,0], :].T
print(bfm_uv['UV'].shape)
print(model_info['trimIndex'].shape)
print(uv_coords.shape)

# modify bad vers
bad_ind = [10032, 10155, 10280]
round1 = [10033, 10158]
round2 = [10534, 10157, 10661]
round3 = [10916, 10286]
uv_coords[:, bad_ind[0]] = np.mean(uv_coords[:, round1], axis=1)
uv_coords[:, bad_ind[1]] = np.mean(uv_coords[:, round2], axis=1)
uv_coords[:, bad_ind[2]] = np.mean(uv_coords[:, round3], axis=1)

model_info['uv_coords'] = uv_coords.T
bfm_uv['UV'] = model_info['uv_coords']

# modify tri mouth
tm_inner = model['tri_mouth']
tm_inner_add = np.array([[6420, 6542, 6664],  # add inner mouth triangles
                         [6420, 6294, 6167],
                         [6167, 6297, 6420],
                         [6167, 6297, 6296],
                         [6167, 6296, 6295],
                         [6167, 6295, 6039],
                         [6168, 6295, 6039]])
ind_bad = 38
print(tm_inner.shape)
all_ind = np.arange(0, tm_inner.shape[1])
tm_inner = np.delete(tm_inner, np.setdiff1d(all_ind, bad_ind), axis=1)
tm_inner = np.concatenate((tm_inner, tm_inner_add.T), axis=1)
model_info['tri_mouth'] = tm_inner
model['tri_mouth'] = tm_inner

# save
sio.savemat('Out/BFM.mat', model)
sio.savemat('Out/BFM_info.mat', model_info)
sio.savemat('Out/BFM_UV.mat', bfm_uv)
shutil.copyfile('3ddfa/pncc_code.mat', 'Out/pncc_code.mat')

