import numpy as np
import scipy.io as sio
from scipy.io import savemat, loadmat
from cmc_plot import cmc_figure

#------------------------------------rank-10 CMC plot-------------------------------------
#euclid_rankk=np.load('euclid_rankk.npy')
#corre_rankk=np.load('corre_rankk.npy')
#cheby_rankk=np.load('cheby_rankk.npy')
#manhat_rankk=np.load('manhat_rankk.npy')
#minkow_rankk=np.load('minkow_rankk.npy')
#cosine_rankk=np.load('cosine_rankk.npy')
#sqeuclid_rankk=np.load('sqeuclid_rankk.npy')
#
#rankk_dict={'euclid':euclid_rankk.T,'correl':corre_rankk.T,'cheby':cheby_rankk.T,'manhattan':manhat_rankk.T,'cosine':cosine_rankk.T,'minkow':minkow_rankk.T,'sq.euclid':sqeuclid_rankk.T}
#sio.savemat('rankk.mat',rankk_dict)
#mat = loadmat('rankk.mat')
#keys = ['euclid', "correl", 'cheby', 'manhattan', 'cosine', 'minkow', "sq. euclid"]
#methods = {"euclid": mat['euclid'],
#           "correl": mat['correl'],
#           "cheby": mat['cheby'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkow": mat['minkow'],
#           "sq. euclid" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



#----------------------------------rank-100 CMC plot--------------------------------------
#euclid_rankk_100=np.load('euclid_rankk_100.npy')
#corre_rankk_100=np.load('corre_rankk_100.npy')
#cheby_rankk_100=np.load('cheby_rankk_100.npy')
#manhat_rankk_100=np.load('manhat_rankk_100.npy')
#minkow_rankk_100=np.load('minkow_rankk_100.npy')
#cosine_rankk_100=np.load('cosine_rankk_100.npy')
#sqeuclid_rankk_100=np.load('sqeuclid_rankk_100.npy')
#
#rankk_dict_100={'euclid':euclid_rankk_100.T,'correl':corre_rankk_100.T,'cheby':cheby_rankk_100.T,'manhattan':manhat_rankk_100.T,'cosine':cosine_rankk_100.T,'minkow':minkow_rankk_100.T,'sq.euclid':sqeuclid_rankk_100.T}
#sio.savemat('rankk_100.mat',rankk_dict_100)
#mat = loadmat('rankk_100.mat')
#keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
#methods = {"euclidean": mat['euclid'],
#           "correlation": mat['correl'],
#           "chebyshev": mat['cheby'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkowski": mat['minkow'],
#           "sq. euclidean" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



# -------------------------------rankk mean CMC plot--------------------------------------
#euclid_mean_cmc=np.load('euclid_mean_cmc.npy')
#corre_mean_cmc=np.load('corre_mean_cmc.npy')
#cheby_mean_cmc=np.load('cheby_mean_cmc.npy')
#manhat_mean_cmc=np.load('manhat_mean_cmc.npy')
#minkow_mean_cmc=np.load('minkow_mean_cmc.npy')
#cosine_mean_cmc=np.load('cosine_mean_cmc.npy')
#sqeuclid_mean_cmc=np.load('sqeuclid_mean_cmc.npy')
#rankk_dict_mean_cmc={'euclidean':euclid_mean_cmc.T,'correlation':corre_mean_cmc.T,'chebyshev':cheby_mean_cmc.T,'manhattan':manhat_mean_cmc.T,'minkowski':minkow_mean_cmc.T,'cosine':cosine_mean_cmc.T,'sq.euclid':sqeuclid_mean_cmc.T}
#sio.savemat('rankk_mean_cmc.mat',rankk_dict_mean_cmc)
#mat = loadmat('rankk_mean_cmc.mat')
#keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
#methods = {"euclidean": mat['euclidean'],
#           "correlation": mat['correlation'],
#           "chebyshev": mat['chebyshev'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkowski": mat['minkowski'],
#           "sq. euclidean" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



# -------------------------------rankk mean CMC plot--------------------------------------
euclid_mean_cmc_100=np.load('euclid_mean_cmc_100.npy')
corre_mean_cmc_100=np.load('corre_mean_cmc_100.npy')
cheby_mean_cmc_100=np.load('cheby_mean_cmc_100.npy')
manhat_mean_cmc_100=np.load('manhat_mean_cmc_100.npy')
minkow_mean_cmc_100=np.load('minkow_mean_cmc_100.npy')
cosine_mean_cmc_100=np.load('cosine_mean_cmc_100.npy')
sqeuclid_mean_cmc_100=np.load('sqeuclid_mean_cmc_100.npy')
rankk_dict_mean_cmc_100={'euclidean':euclid_mean_cmc_100.T,'correlation':cosine_mean_cmc_100.T,'chebyshev':cheby_mean_cmc_100.T,'manhattan':manhat_mean_cmc_100.T,'minkowski':minkow_mean_cmc_100.T,'cosine':cosine_mean_cmc_100.T,'sq.euclid':sqeuclid_mean_cmc_100.T}
sio.savemat('rankk_mean_cmc_100.mat',rankk_dict_mean_cmc_100)
mat = loadmat('rankk_mean_cmc_100.mat')
keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
methods = {"euclidean": mat['euclidean'],
           "correlation": mat['correlation'],
           "chebyshev": mat['chebyshev'],
           "manhattan": mat['manhattan'],
           "cosine": mat['cosine'],
           "minkowski": mat['minkowski'],
           "sq. euclidean" : mat['sq.euclid']}
cmc_figure(methods, keys=keys)



# --------------------------------------mAP CMC plot---------------------------------------
#euclid_map=np.load('euclid_map.npy')
#corre_map=np.load('corre_map.npy')
#cheby_map=np.load('cheby_map.npy')
#manhat_map=np.load('manhat_map.npy')
#minkow_map=np.load('minkow_map.npy')
#cosine_map=np.load('cosine_map.npy')
#sqeuclid_map=np.load('sqeuclid_mean_cmc.npy')
#rankk_dict={'euclidean':euclid_mean_cmc.T,'correlation':corre_mean_cmc.T,'chebyshev':cheby_mean_cmc.T,'manhattan':manhat_mean_cmc.T,'minkowski':minkow_mean_cmc.T,'cosine':cosine_mean_cmc.T,'sq.euclid':sqeuclid_mean_cmc.T}
#sio.savemat('rankk.mat',rankk_dict)
#mat = loadmat('rankk.mat')
#keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
#methods = {"euclidean": mat['euclidean'],
#           "correlation": mat['correlation'],
#           "chebyshev": mat['chebyshev'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkowski": mat['minkowski'],
#           "sq. euclidean" : mat['sq. euclidean']}
#cmc_figure(methods, keys=keys)