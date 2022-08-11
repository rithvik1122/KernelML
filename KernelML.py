import sys
import numpy as np
import pandas as pd
from PyAstronomy import pyasl
np.set_printoptions(threshold=sys.maxsize)

def Mask(LengthTotal,LengthTrue):
    mask = np.full(LengthTotal, False)
    mask[:LengthTrue] = True
    np.random.shuffle(mask)
    return mask

def RepresentationMatrix(AtomDataTrain):

    VRC = []
    y = []
    for k in range(len(AtomDataTrain)):
        Mij = np.zeros((len(AtomDataTrain[k]),len(AtomDataTrain[k])))
        y.append(AtomDataTrain[k][0][1])
        for i in range(len(AtomDataTrain[k][1:])):
            for j in range(len(AtomDataTrain[k][1:])):
                Zi = pyasl.AtomicNo().getAtomicNo(AtomDataTrain[k][1:][i][0])
                Zj = pyasl.AtomicNo().getAtomicNo(AtomDataTrain[k][1:][j][0])
                if i==j:
                    Mij[i][j]=0.5*(Zi*Zj)**(2.4)

                else:            
                    R1 = AtomDataTrain[k][1:][i][1:]
                    R2 = AtomDataTrain[k][1:][j][1:]
                    Norm = np.linalg.norm(R1-R2)
                    Mij[i][j] = Zi*Zj/Norm

        SortList = np.argsort(np.linalg.norm(Mij,axis=0)) # sorted according to norm
        MijSorted = Mij[SortList]

        NZ = 23-len(AtomDataTrain[k][1:]) # no. of rows and column we want to extend our matrix by
        Mij_New = np.lib.pad(MijSorted,(0,NZ), 'constant', constant_values=(0))

        VectorRep = Mij_New[np.tril_indices(len(Mij_New),k=0)] # vector representation of lower triangular part,
                                                               # including the diagonal of Mij_New

        VRC.append(VectorRep)
        
    return VRC,y

def KERNEL(VRCTrain,VRCHoldOut,Sig):
    Kernel = np.zeros([len(VRCTrain),len(VRCHoldOut)])
    for i in range(len(VRCTrain)):
        for j in range(len(VRCHoldOut)):
            if len(VRCTrain)==len(VRCHoldOut):
                if i<j:
                    Norm = np.linalg.norm(VRCTrain[i]-VRCHoldOut[j])**2
                    Kernel[i][j] = np.exp(-Norm/(2*Sig**2)) # Upper Triangular Gaussian Kernel Matrix
            else:
                    Norm = np.linalg.norm(VRCTrain[i]-VRCHoldOut[j])**2
                    Kernel[i][j] = np.exp(-Norm/(2*Sig**2)) # Upper Triangular Gaussian Kernel Matrix

    if len(VRCTrain)==len(VRCHoldOut):
        Kernel = Kernel + Kernel.T - np.diag(Kernel.diagonal()) # Complete Kernel Matrix
    
    return Kernel

