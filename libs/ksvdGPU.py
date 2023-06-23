# coding:utf-8
import numpy as np
import scipy as sp
import torch
from PursuitMethods import OMP as OMP


class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements

        max_iter:
            Maximum number of iterations

        tol:
            tolerance for error

        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        if torch.cuda.device_count()==0:
            torch.device('cpu')
            self.device = 'cpu'
        else:
            torch.device('cuda:0')
            self.device = 'cuda:0'
        

        #if type(n_components)=="numpy.ndarray":
        #    #Pasar a tensor
        #elif type(n_components)=="torch.Tensor":
        #    print("Cant inicialize becose n_components type is wrong. Type: ",type(n_components))

        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.misshistory = None

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if torch.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)        # <------ CONTROLAR ACÁ que haga el producto bien
            d = r.T.dot(g)                          # <------ CONTROLAR ACÁ que haga el producto bien
            d /= torch.linalg.norm(d)
            g = r.dot(d)                            # <------ CONTROLAR ACÁ que haga el producto bien
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X, initrand = False):
        if initrand==False:
            D = torch.randn(self.n_components, X.size()[1])
            D /= torch.linalg.norm(D, axis=1)[:, np.newaxis]
            return D

        if X.size()[0] >= self.n_components:
            ind = torch.randperm(X.size(0))
            D = X[ind[:self.n_components]]
        else:
            #u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            u, s, vt = torch.linalg.svd(X, full_matrices=False)
            s = s[:self.n_components]
            vt = vt[:self.n_components]                     # <-------------------------
            D = torch.matmul(torch.diag(s), vt)        # <-------------- controlar que de ka multiplicacion
        D /= torch.linalg.norm(D, axis=1)[:, np.newaxis]    # <-------------- Debuguear y ver que dimensiones da
        return D

       
    def _transform(self, D, X):
        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.size()[1])   # <-------------- Probar si funciona sin parentesis
        return OMP(X, D,max_iterations=n_nonzero_coefs)
         
    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        if type(X)=="numpy.ndarray":
            X = torch.from_numpy(X)
        elif type(X)=="torch.Tensor":
            print("Cant inicialize becose n_components type is wrong. Type: ",type(X))
        self.misshistory = []
        #### ---------------------------
        D = self._initialize(X)


        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = torch.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)
            miss = self.comperror(X,D,gamma)
            self.misshistory.append(miss)
            print("Iteracion " + str(i+1) +"/"+str(self.max_iter) + ", RMSE = "+ str(miss))
        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)
    
    def comperror(self,data,dicc,gamma):
        #Calculo del error por RMSE
        return torch.sqrt(torch.sum( (data-dicc.matmul(gamma))**2 ))/data.size(dim=0)
        


#import tensorly as tl
#tl.set_backend('pytorch')

#U, S, V = tl.truncated_svd(matrix, n_eigenvecs=10)