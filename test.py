import scipy.io as sio

# model = sio.loadmat("data/Out/BFM.mat")

# print(model.keys())

# print(model['model'].shape)

# print(type(model['model'][0,0]))
# print(len(model['model'][0,0]))
# print(model['model'][0,0][0].shape)
# print(model['model'][0,0][1].shape)
# print(model['model'][0,0][2].shape)
# print(type(model['model']))



# model_infor = sio.loadmat("data/Out/BFM_info.mat")

# print(model_infor.keys())
# print(type(model_infor['model_info']))
# print(model_infor['model_info'].shape)

import numpy as np 

# A = np.array(
#     [
#         [1,2,3,4],
#         [5,6,7,8],
#         [9,10,11,12],
#     ]
# )
# A = np.array(
#     [[1,2,3,4]]
# ).T
# print(A.shape)

# B = np.concatenate([3 * A - 2, 3 * A -1, 3 * A], axis= 1).T
# print(B)
# print(B.flatten(order='F'))

# trimIndex_f_target =  sio.loadmat("data/trimIndex_f.mat")['trimIndex_f']
# print(trimIndex_f_target['trimIndex_f'])



# UV_mat = sio.loadmat('data/Out/BFM_info.mat')['model_info']
# print(type(UV_mat))
# print(UV_mat.shape)


# BFM_model_test = sio.loadmat("BFM_test.mat")['model'][0,0]

# print(len(BFM_model_test))



BFM = sio.loadmat("BFM.mat")
print(type(BFM))
print(((BFM['model'][0,0]['shapeMU'].shape)))
# print((BFM['model'][0,0]))