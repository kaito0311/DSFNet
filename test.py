import scipy.io as sio

bfm_model = sio.loadmat('300W_LP/AFW_Flip/AFW_134212_1_0.mat')

print(bfm_model.keys())