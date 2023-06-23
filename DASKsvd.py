import numpy as np
import torch as tc
#from libs.ksvdGPU import ApproximateKSVD as KsvdGPU
from libs.ksvdCPU import ApproximateKSVD as KsvdGPU # passtrick
from libs.ksvdCPU import ApproximateKSVD as KsvdCPU
from libs.PursuitMethods import OMP as OMP
from math import ceil, floor
import spams


class diccionariospams:
    def __init__(self,q):
        self.param = { 'K' : q, # learns a dictionary with 100 elements
          'lambda1' : 0.15, 'numThreads' : 4, 'batchsize' : 400,
          'iter' : 500, 'verbose': True}
    
    def fit(self,X):
        return spams.trainDL(X, **self.param)
        


#data_train,labels_train,'q',sparsity,'r',r,'discMeasureParams',parameters
class DASKsvd:
    '''
    X : Datos de entrada para entrenamiento, dispuesto en forma de "samples x features"
        Ej: 200 señales de 128 muestras de longitud (200x128)
    
    Y : Labels, Targets o clases de los datos en X, vector o lista de cantidad igual a los samples
        Ej: vector o lista de 200 elementos correspondinte a las 200 señales (unidimensional)
    
    p_percent: porcentaje de elementos por clase utilizados para el entrenamiento en cada iteracion
                el porcentaje da un valor en relacion a la clase de menos frecuente

    sparsity : proporcion de "sparsidad" de las representaciones optenidas, es decir componentes no nulos
            debe estar entre  0 < q < 1    
    
    redundance : define la cantidad de atomos del diccionario en funcion a la longitud de las señales
        Ej: Si se define un factor de redundancia de 2 y la señal 128 features, el diccionario tendrá     
        un tamaño de 256 atomos X 128

    ItDas : Cantidad de iteraciones del DAS-Ksvd, en funcion de la cantidad de iteraciones genera un diccionario con "ItDAS" atomos por clase
        Si son 3 clases el atomo (ItDas X clase)
    
    ItKsvd : Iteraciones del metodo Ksvd
    
    pdescuento : disminucion de la distribucion de probabilidad para los samples seleccionados en las iteraciones del DAS
    
    addnoice : Ruido adicionado a los datos para generalizar, se adiciona en cada iteracion del DAS
    
    discMeasureParams : Define los coeficientes que acompañan al metodo de discriminacion de atomos
     deben ser pares de valores cuya suma sea <=1. Ej: [0.8, 0.1] o matrices [ [1, 0] [0.1, 0.5] [0.8, 0.2] ]
     Para mas detalle analizar los metodos de discriminacion MDAS y MDCS

    Las funciones de torch y numpy se utilizan segun el flag torchuse al inicializar
    '''


    def __init__(self, X, Y,p_percent ,sparsity,redundance,ItDas = 20, ItKsvd=15,pdescuento=0.5,addnoice=0.1 ,discMeasureParams = [[1,0]],torchuse=False,verbose=False):
        self.X = X
        self.targets = Y
        self.ntargets = None 
        self.q = sparsity
        self.r = redundance
        self.DiscCoeficient = discMeasureParams
        self.beta = pdescuento          # porcentaje de descuento de la distribucion de probabilidad
        self.desvnoice = addnoice       # Desviacion de ruido agregada a la señal
        self.Itdas = ItDas
        self.ItKsvd = ItKsvd
        self.probdiscont = pdescuento
        self.noicecoef = addnoice
        self.D_Phi = None
        
        if torchuse and tc.cuda.device_count()!=0:
            tc.device('cuda:0')
            self.ntargets = tc.unique(self.targets)
            self.countclass = []
            for itclass in self.ntargets:
                self.countclass.append((tc.sum(Y==itclass)).item())
            self.countclass = tc.Tensor(self.countclass)
            self.ndataxit = int(tc.floor(tc.min(self.countclass)* p_percent).item()) 
            self.natoms = self.r * X.size(dim=1)
            self.ompmetohd = OMP

            self.selecciondatos = self.sampledataGPU
            self.ksvd = KsvdGPU
            self.discrimatoms = self.DiscAtomsGPU    
            self.device = 'cuda:0'
            self.mainprocess = self.mainprocessGPU
        else:
            tc.device('cpu')
            self.ntargets = np.unique(self.targets)          
            self.countclass = np.array([],dtype=np.int16)
            for itclass in self.ntargets:
                self.countclass = np.append(self.countclass,np.sum(Y==itclass))
            self.ndataxit = int(np.floor(np.min(self.countclass)* p_percent)) 
            self.natoms = self.r * X.shape[1]
            self.ompmetohd = self.omp

            self.selecciondatos = self.sampledataCPU
            self.ksvd = KsvdCPU
            self.discrimatoms = self.DiscAtomsCPU
            self.device = 'cpu'
            self.mainprocess = self.mainprocessCPU

        # Setting Ksvd method       
        self.dlmethod = self.ksvd(self.natoms,self.ItKsvd,transform_n_nonzero_coefs= ceil(self.natoms * self.q),verbose=verbose)
        #self.dlmethod = diccionariospams(self.natoms)
        #### CAMBIAR LINEAS CON "dlmethod" en lineas 148 y 149 y 183 y 184
        self.mainprocess()

        return

    def omp(self,X,D,n_nonzero_coefs):
        return spams.omp(X.T,D.T,L=n_nonzero_coefs).toarray().T

    def randomselectorch(self):
        # Despreciado por tiempo de procesamiento, es poco usado asi que a consideracion no vale la pena
        return

    def sampledataGPU(self,probin):
        # Retorna 2 vectores, el primero de los indices para los datos
        # y el segundo con la distribucion de probabilidad reducida en el factor dado
        # para la seleccion
        probout = probin.copy()
        numbers = np.arange(probin.size)
        ncl = self.ntargets.size(0)
        selected = np.zeros(self.ndataxit * ncl,dtype=np.int32)
        for i in range(ncl):
            selected[i * self.ndataxit: (i+1)*self.ndataxit] = np.random.choice(numbers[self.targets==self.ntargets[i]],self.ndataxit,False,p=probin[self.targets==self.ntargets[i]]/probin[self.targets==self.ntargets[i]].sum())
        probout[selected] = probin[selected] * self.probdiscont
        return selected, probout

    def sampledataCPU(self,probin):
            # Retorna 2 vectores, el primero de los indices para los datos
            # y el segundo con la distribucion de probabilidad reducida en el factor dado
            # para la seleccion
            probout = probin.copy()
            numbers = np.arange(probin.size)
            ncl = self.ntargets.size
            selected = np.zeros(self.ndataxit * ncl,dtype=np.int32)
            for i in range(ncl):
                selected[i * self.ndataxit: (i+1)*self.ndataxit] = np.random.choice(numbers[self.targets==self.ntargets[i]],self.ndataxit,False,p=probin[self.targets==self.ntargets[i]]/probin[self.targets==self.ntargets[i]].sum())
            probout[selected] = probin[selected] * self.probdiscont
            return selected, probout 

    def mainprocessGPU(self):
        elems, m = self.X.size()
        probabilitydist = np.ones(elems) / elems

        numgrilla = len(self.DiscCoeficient)
        if numgrilla<2:
            DPhi = tc.zeros((1,self.Itdas*self.ntargets.size()[0],m),device=self.device)
        else:
            DPhi = tc.zeros((numgrilla,self.Itdas*self.ntargets.size()[0],m),device=self.device)


        for itdas in range(self.Itdas):
            indices, probabilitydist = self.selecciondatos(probabilitydist)
            # Aprendizaje por Ksvd
            self.dlmethod.fit(self.X[indices]) 
            Phi = self.dlmethod.components_
            #Phi = (self.dlmethod.fit(self.X[indices].T)).T
            
            # Obtencion de la representacion de los datos
            gamma = self.ompmetohd(self.X[indices],Phi, ceil(self.natoms*self.q) )
            # Algoritmo de discriminacion
            [coefDisc, bestclass, secondclass] = self.DiscAtomsGPU(self.X[indices],gamma,Phi,self.targets[indices],self.DiscCoeficient)
            # Extraccion del atomo por clase
            bestclass = tc.Tensor(bestclass)
            for gridit in range(numgrilla):
                for itcl,namecl in enumerate(self.ntargets):
                    #Seleccion de atomos por clase
                    seleccion = (bestclass==itcl)
                    item =   (coefDisc[gridit]*seleccion).argmax()
                    DPhi[gridit,itcl*self.Itdas+itdas] = item
        self.D_Phi = DPhi
        return DPhi
       

    def mainprocessCPU(self):
        elems, m = self.X.shape
        probabilitydist = np.ones(elems) / elems

        numgrilla = len(self.DiscCoeficient)
        if numgrilla<2:
            DPhi = np.zeros((1,self.Itdas*self.ntargets.size,m))
        else:
            DPhi = np.zeros((numgrilla,self.Itdas*self.ntargets.size,m))


        for itdas in range(self.Itdas):
            print("Iteracion DAS-KSVD "+ str(itdas+1) +"/"+str(self.Itdas))
            indices, probabilitydist = self.selecciondatos(probabilitydist)
            # Aprendizaje por Ksvd
            self.dlmethod.fit(self.X[indices]) 
            Phi = self.dlmethod.components_
            #Phi = (self.dlmethod.fit(self.X[indices].T)).T

            # OMP con matriz de gram
            #gamma = self.ompmetohd( Phi.dot(Phi.T) , Phi.dot(self.X[indices].T) ,n_nonzero_coefs=ceil(self.natoms*self.q) ).T
            #OMP de spam
            gamma = self.ompmetohd( self.X[indices], Phi, n_nonzero_coefs=ceil(self.natoms*self.q))
            # Algoritmo de discriminacion
            [coefDisc, bestclass, secondclass] = self.DiscAtomsCPU(self.X[indices],gamma,Phi,self.targets[indices],self.DiscCoeficient)
            # Extraccion del atomo por clase
            bestclass = np.array(bestclass)
            for gridit in range(numgrilla):
                for itcl,namecl in enumerate(self.ntargets):
                    #Seleccion de atomos por clase
                    seleccion = (bestclass==itcl)
                    item =  (coefDisc[gridit]*seleccion).argmax()
                    DPhi[gridit,itcl*self.Itdas+itdas] = Phi[item]
        self.D_Phi = DPhi
        return DPhi


    
    def DiscAtomsCPU(self,datos,represent,dicc,labels,parametros):
        clases = np.unique(labels)
        ncl = clases.shape[0]

        priorclass = []
        secondclass= []

        errorinclass = np.zeros(ncl,dtype=object)
         
        for itcl in range(ncl):
            segment = labels == self.ntargets[itcl]
            errorinclass[itcl] = datos[segment]-represent[segment].dot(dicc)

        ###
        # -- Frecuencia de activacion
        ###
        
        m_af = np.zeros(dicc.shape[0])
        for atm in range(dicc.shape[0]):
            probs = []
            for itcl in range(ncl):
                indsubset = (labels == clases[itcl])
                subset = represent[indsubset,atm]
                probs.append( (subset!=0).sum() / indsubset.sum() )
            probs = np.array(probs)
            [c1, c2] = np.argsort(probs)[:-3:-1]
            priorclass.append(c1)
            secondclass.append(c2)
            m_af[atm] = (probs[c1]-probs[c2])/probs[c1]
        
        m_cm = np.zeros(dicc.shape[0])
        m_re = np.zeros(dicc.shape[0])
        for atm in range(dicc.shape[0]):
            ind1 = self.ntargets[priorclass[atm]]==labels 
            ind2 = self.ntargets[secondclass[atm]]==labels 
            nind1 = np.array(ind1).sum()
            nind2 = np.array(ind2).sum()
        
            ###
            # -- Magnitud de coeficiente
            ###
                
            # Calculo de los coeficientes para las mejores clases encontradas antes
            auxcoefprior = np.sum(np.abs( represent[ind1,atm]))/nind1
            auxcoefsecond= np.sum(np.abs( represent[ind2,atm]))/nind2
            if auxcoefprior > auxcoefsecond:
                m_cm[atm] = (auxcoefprior-auxcoefsecond)/auxcoefprior

            ###
            # -- Error de representacion
            ###
            auxerrprior = np.linalg.norm(errorinclass[priorclass[atm]]+ np.matmul(represent[ind1,atm].reshape(-1,1),dicc[atm].reshape(1,-1)),'fro').sum() / nind1 
            auxerrsecon = np.linalg.norm(errorinclass[secondclass[atm]]+ np.matmul(represent[ind2,atm].reshape(-1,1),dicc[atm].reshape(1,-1)),'fro').sum()/ nind2
            if auxcoefprior > auxcoefsecond:
                m_re[atm] = (auxerrprior-auxerrsecon)/auxerrprior

        comp = []
        for param in parametros:
            comp.append( param[0]*m_af + param[1]*m_cm + (param[0]-param[1])*m_re)
        
        return np.array(comp), priorclass, secondclass
      



    def DiscAtomsGPU(self,datos,represent,dicc,labels,parametros):
        clases = tc.unique(labels)
        ncl = clases.size()
        
        priorclass = []
        secondclass= []
        
        '''errorinclass = tc.zeros(ncl,dtype=object)
        for itcl in range(ncl):
            segment = labels == self.ntargets[itcl]
            errorinclass[itcl] = datos[segment]-represent[segment].dot(dicc)
        '''
        errortotal = datos-represent.matmul(dicc)
        ###
        # -- Frecuencia de activacion
        ###
        m_af = tc.zeros(dicc.size(dim=0))
        
        for atm in range(dicc.size(dim=0)):
            probs = []
            for itcl in range(ncl):
                indsubset = (labels == clases[itcl])
                subset = datos[indsubset,atm]
                probs.append( (subset!=0).sum() / indsubset.sum() )
            probs = np.array(probs)
            [c1, c2] = np.argsort(probs)[:-3:-1]
            priorclass.append(c1)
            secondclass.append(c2)
            m_af.append(probs[c1]-probs[c2])/probs[c1]
        
        m_cm = tc.zeros(dicc.size(dim=0))
        m_re = tc.zeros(dicc.size(dim=0))
        for atm in range(dicc.size(dim=0)):
            ind1 = self.ntargets[priorclass[atm]]==labels 
            ind2 = self.ntargets[secondclass[atm]]==labels
            nind1 = np.array(ind1).sum()
            nind2 = np.array(ind2).sum()
        ###
        # -- Magnitud de coeficiente
        ###
            auxcoefprior = tc.sum(tc.abs(represent[ind1,atm]))/nind1
            auxcoefsecond= tc.sum(tc.abs(represent[ind2,atm]))/nind2
            if auxcoefprior > auxcoefsecond:
                m_cm[atm] = (auxcoefprior-auxcoefsecond)/auxcoefprior
        ###
        # -- Error de representacion
        ###
            # El ,sum()
            
            auxerrprior = tc.linalg.norm(errortotal[labels==priorclass[atm]]+ represent[ind1,atm].matmul(dicc[atm]),ord='fro').sum() / nind1 
            auxerrsecon = tc.linalg.norm(errortotal[labels==secondclass[atm]]+ represent[ind2,atm].matmul(dicc[atm]),ord='fro').sum()/ nind2
            if auxcoefprior > auxcoefsecond:
                m_re[atm] = (auxerrprior-auxerrsecon)/auxerrprior
        
        comp = []
        for param in parametros:
            comp.append( param[0]*m_af + param[1]*m_cm + (param[0]-param[1])*m_re)
        
        return tc.Tensor(comp), priorclass, secondclass

        


