#!/usr/bin/env python


import torch
import numpy as np
#from scipy.spatial.distance import cdist

#datadir = '/Users/ksargsy/GDrive/Work/run/surr_int/h_3d/'


kB = 8.6173e-5  # eV/K
m = 1.6735575e-27
h = 6.62607004 * 1.e-34  # m2 kg / s
ev_to_j = 1.60218e-19  # J/K

indperm = np.array([2, 0, 1], dtype=int)
indperm_inv = np.array([1, 2, 0], dtype=int) # z,x,y to x,y,z

zmin, zmax = 0.41748, 4.41358

# fcc:
center1 = np.array([1.26572824, 0.73075499, 0.91748354])
hess1 = np.array([[2.73218277e+00, 8.99886106e-05, -2.44254800e-03],
                  [8.99886106e-05, 2.73219563e+00, -7.13481127e-04],
                  [-2.44254800e-03, -7.13481127e-04, 4.14538963e+00]])
yshift1 = 0.0

# hcp:
center2 = np.array([2.53145314, 1.4615272, 0.91993214])
hess2 = np.array([[2.68536299e+00, -2.57110316e-05, -5.14220632e-05],
                  [-2.57110316e-05, 2.68522158e+00, 2.05688253e-04],
                  [-5.14220632e-05, 2.05688253e-04, 4.16232034e+00]])
yshift2 = 0.000466675235657

# bridge:
center3 = np.array([0.00000119, -0.00000067, 1.51315333])
hess3 = np.array([[4.75354551e+00, -6.42775790e-05, 2.12116011e-04],
                  [-6.42775790e-05, -8.22380202e-01, -1.78048894e-03],
                  [2.12116011e-04, -1.78048894e-03, 5.24415056e+00]])
yshift3 = 0.589240068221

# atop:
center4 = np.array([1.26572984, -0.00251192, 1.0616334])
hess4 = np.array([[-4.83945893e-01, -2.31399285e-04, -3.85665474e-05],
                  [-2.31399285e-04, -4.83997315e-01, -2.12116011e-04],
                  [-3.85665474e-05, -2.12116011e-04, 1.12360551e+01]])
yshift4 = 0.133963275468


centers = [center1, center2]  # , center3, center4]
hessians = [hess1, hess2]  # , hess3, hess4]
yshifts = [yshift1, yshift2]  # , yshift3, yshift4]




x0, y0, z0 = 0, 0, zmin
init = np.array([x0, y0, z0]).reshape(-1, 3)
delta_x = 7.594326829943521 / 3.
delta_y = delta_x * np.sqrt(3.) / 2   # 6.576879959372833 /3.
delta_z = zmax - zmin
rh_transform = np.eye(3) * np.diag(np.array([delta_x, delta_y, delta_z]))
rh_transform[1, 0] = delta_x / 2.
invrh_transform = np.linalg.inv(rh_transform)


def quad(x, hess, center):
    nsam, ndim = x.shape
    assert(ndim == hess.shape[0])
    assert(ndim == hess.shape[1])
    assert(ndim == center.shape[0])

    yy = np.empty(nsam,)
    for i in range(nsam):
        yy[i] = 0.5 * np.dot(x[i, :] - center, np.dot(hess, x[i, :] - center))

    return yy


def quad_MCint(x, hess, center):
    #nsam, ndim = x.shape
    ndim=x.shape[0]
    #print(ndim)
    nsam=1
    assert(ndim == hess.shape[0])
    assert(ndim == hess.shape[1])
    assert(ndim == center.shape[0])

    yy = np.empty(nsam,)
    if nsam>1:
        for i in range(nsam):
            yy[i] = 0.5 * np.dot(x[i, :] - center, np.dot(hess, x[i, :] - center))
    else:
        yy = 0.5 * np.dot(x[:] - center, np.dot(hess, x[:] - center))
    return yy


def eval_energy_joint(xtest, models, eps=0.2):

    ntst = 1 #xtest.shape[0]
    ypred_joint = np.zeros(ntst,)

#    dists = cdist(xtest, np.array(centers))
    if len(xtest)==3:
        dists=[np.linalg.norm(xtest-np.array(center1)),np.linalg.norm(xtest-np.array(center2))]
        dists=np.array(dists)
    else: #if more than one point
        dists=np.zeros((len(xtest[:,0]),2))
        for xi in range(len(xtest[:,0])):
            dists[xi,0]=np.linalg.norm(xtest[xi,:]-np.array(center1))
            dists[xi,1]=np.linalg.norm(xtest[xi,:]-np.array(center2))

    scales = np.exp(-dists / eps)
    #scales /= np.sum(scales, axis=1).reshape(-1, 1) #when there are more points
    scales /= np.sum(scales)

    ncl = len(centers)
    assert(ncl == len(hessians))

    for j in range(ncl):
        center = centers[j]
        hess = hessians[j]
        yshift = yshifts[j]
        model = models[j]

        ypred_ = model(torch.from_numpy(xtest - center).double()).detach().numpy().reshape(-1,)

        ypred = yshift + quad_MCint(xtest, hess, center) * np.exp(ypred_)

        ypred_joint += ypred * scales[j] #[:, j] include if more points

    return ypred_joint
