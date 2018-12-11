import numpy as np
import scipy.io as sio
from scipy.io import savemat, loadmat
from cmc_plot import cmc_figure

# rankk CMC plot
euclid_rankk=np.load('euclid_rankk.npy')
corre_rankk=np.load('corre_rankk.npy')
cheby_rankk=np.load('cheby_rankk.npy')
manhat_rankk=np.load('manhat_rankk.npy')
minkow_rankk=np.load('minkow_rankk.npy')
cosine_rankk=np.load('cosine_rankk.npy')
sqeuclid_rankk=np.load('sqeuclid_rankk.npy')

rankk_dict={'euclid':euclid_rankk.T,'correl':corre_rankk.T,'cheby':cheby_rankk.T,'manhattan':manhat_rankk.T,'cosine':cosine_rankk.T,'minkow':minkow_rankk.T,'sq.euclid':sqeuclid_rankk.T}
sio.savemat('rankk.mat',rankk_dict)
mat = loadmat('rankk.mat')
keys = ['euclid', "correl", 'cheby', 'manhattan', 'cosine', 'minkow', "euclid"]
methods = {"euclid": mat['euclid'],
           "correl": mat['correl'],
           "cheby": mat['cheby'],
           "manhattan": mat['manhattan'],
           "cosine": mat['cosine'],
           "minkow": mat['minkow'],
           "euclid" : mat['sq.euclid']}
cmc_figure(methods, keys=keys)



# rankk mean CMC plot
euclid_mean_cmc=np.load('euclid_mean_cmc.npy')
corre_mean_cmc=np.load('corre_mean_cmc.npy')
cheby_mean_cmc=np.load('cheby_mean_cmc.npy')
manhat_mean_cmc=np.load('manhat_mean_cmc.npy')
minkow_mean_cmc=np.load('minkow_mean_cmc.npy')
cosine_mean_cmc=np.load('cosine_mean_cmc.npy')
sqeuclid_mean_cmc=np.load('sqeuclid_mean_cmc.npy')
rankk_dict={'euclidean':euclid_mean_cmc.T,'correlation':corre_mean_cmc.T,'chebyshev':cheby_mean_cmc.T,'manhattan':manhat_mean_cmc.T,'minkowski':minkow_mean_cmc.T,'cosine':cosine_mean_cmc.T,'sq.euclid':sqeuclid_mean_cmc.T}
sio.savemat('rankk.mat',rankk_dict)
mat = loadmat('rankk.mat')
keys = ['euclid', "correl", 'cheby', 'manhattan', 'cosine', 'minkow', "euclid"]
methods = {"euclidean": mat['euclid'],
           "correl": mat['correl'],
           "cheby": mat['cheby'],
           "manhattan": mat['manhattan'],
           "cosine": mat['cosine'],
           "minkow": mat['minkow'],
           "euclid" : mat['sq.euclid']}
cmc_figure(methods, keys=keys)
#