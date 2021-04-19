#!/usr/bin/env python3

import torch
import numpy as np
import pickle as pk
#from scipy.spatial.distance import cdist



# This should replace common?
kB = 8.6173e-5  # eV/K
m = 1.6735575e-27
h = 6.62607004 * 1.e-34  # m2 kg / s
ev_to_j = 1.60218e-19  # J/K

zmin, zmax = 0.41696773, 5.16696773

def savepk(sobj, nameprefix='savestate'):
    pk.dump(sobj, open(nameprefix + '.pk', 'wb'), -1)


def loadpk(nameprefix='savestate'):
    return pk.load(open(nameprefix + '.pk', 'rb'))

def rel_l2(predictions, targets):
    return np.linalg.norm(predictions - targets) / np.linalg.norm(targets)


class Quadratic():
    def __init__(self, center, hess):
        self.center = center
        self.hess = hess

        return

    def __call__(self, x):
        nsam = 1#x.shape[0]
        #print(nsam)
        yy = np.empty(nsam,)
        for i in range(nsam):
            #yy[i] = 0.5 * np.dot(x[i, :]-self.center, np.dot(self.hess, x[i, :]-self.center)) #use if more than one point
            yy[i] = 0.5 * np.dot(x-self.center, np.dot(self.hess, x-self.center))

        return yy


class MPoint():
    def __init__(self, center, hess, yshift):
        self.center = center
        self.hess = hess
        self.yshift = yshift

        return

class SModel():
    def __init__(self, ptmodel, mpt):
        self.ptmodel = ptmodel
        self.mpt = mpt

        return

    def __call__(self, x):

        quad = Quadratic(self.mpt.center, self.mpt.hess)
        ypred = self.ptmodel(torch.from_numpy(x - self.mpt.center).double()).detach().numpy().reshape(-1,)
        return self.mpt.yshift + quad(x) * np.exp(ypred)


class WFcn():
    def __init__(self, mpts, eps):
        self.mpts = mpts
        self.centers = [mpt.center for mpt in self.mpts]
        self.eps = eps
        return

    def __call__(self, x):
        #dists = cdist(x, np.array(self.centers))
        if len(x)==3:
            #dists=[np.linalg.norm(x-np.array(center1)),np.linalg.norm(x-np.array(center2))]
            dists=np.linalg.norm(x-np.array(self.centers)[None, :, :], axis=2)
            dists=dists[0]
            #print(dists)
        else: #if more than one point
            dists = np.linalg.norm(x[:, None, :] - np.array(self.centers)[None, :,  :], axis=2)
        scales = np.exp(-dists / self.eps)
        #scales /= np.sum(scales, axis=1).reshape(-1, 1)
        return scales


class MultiModel(object):
    def __init__(self, models, wfcn=None, cfcn=None):
        super(MultiModel, self).__init__()
        self.models = models
        self.nmod = len(self.models)

        assert(wfcn is not None or cfcn is not None)

        if wfcn is not None:
            assert(cfcn is None)
            self.wflag = True
            self.wfcn = wfcn
        if cfcn is not None:
            assert(wfcn is None)
            self.wflag = False
            self.cfcn = cfcn



    def __call__(self, x):

        if self.wflag:
            #print(self.wfcn(x))
            val = self.wfcn(x)[0] * self.models[0](x).reshape(-1,)
            #val = self.wfcn(x)[:, 0] * self.models[0](x).reshape(-1,) #use if more than one point
            #summ = self.wfcn(x)[:, 0] #use if more than one point
            summ = self.wfcn(x)[0]
            for j in range(1, self.nmod):
                val += self.wfcn(x)[j] * self.models[j](x).reshape(-1,)
                summ += self.wfcn(x)[j]
                #val += self.wfcn(x)[:, j] * self.models[j](x).reshape(-1,) #use if more than one point
                #summ += self.wfcn(x)[:, j] #use if more than one point
            return val/summ

        else:
            y = np.empty((x.shape[0]))
            for j in np.unique(self.cfcn(x)):
                y[self.cfcn(x)==j]=self.models[j](x[self.cfcn(x)==j, :]).reshape(-1,)
            return y



def eval_mmodel(xtest, ptmodels, mpts, eps=0.2):

    wfcn = WFcn(mpts, eps)
    models = [SModel(ptmodel, mpt) for ptmodel, mpt in zip(ptmodels, mpts)]


    mmodel = MultiModel(models, wfcn=wfcn)

    return mmodel(xtest)
