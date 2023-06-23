import numpy as np
from DASKsvd import DASKsvd
#from DASKsvd import diccionariospams
import torch as tc
import os
PATHscript = os.path.dirname(__file__)+"/"
os.chdir(PATHscript)

namefile = "disc_diccAfro"
datos = np.load(namefile+"-data.npz")
X_train = datos['xtrain']
y_train = datos['ytrain']

modelone = DASKsvd(X_train,y_train,p_percent=0.2,sparsity=0.2, redundance=2,ItDas = 50,ItKsvd=25,discMeasureParams=[[0.7,0.1],[0.8,0.1],[0.75,0.1]],torchuse=False)
DPhi=modelone.D_Phi
for it,dic in enumerate(DPhi):
    np.save(namefile+str(it),dic)
# , 
# ItKsvd=15,
# pdescuento=0.5,addnoice=0.1 
# discMeasureParams = [[1,0]]

#### GPU
#X_train = tc.Tensor(X_train)
#y_train = tc.Tensor(y_train)
#modeltwo = DASKsvd(X_train,y_train,p_percent=0.2,sparsity=0.2, redundance=2,ItDas = 50,ItKsvd=15,discMeasureParams=[[0.7,0.1]])



print("FIN")


