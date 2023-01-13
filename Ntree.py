import numpy as np 
import pylab as py 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from matplotlib import rcParams
import time
import matplotlib
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import StrMethodFormatter
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
matplotlib.rcParams['text.usetex']=True
from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
from sklearn import tree
#######################################################################
ns=int(6)
#head=['mbase','DeltaA',  'FWHM',  'devi', 'Tmax'  , 'rho' , 'tstar' , 'ur' ,  'fb' , 'mstar']
#df= pd.read_csv("./paramall1.txt", sep=" ",  skipinitialspace = True, header=None, usecols=[0,1,2,3,4,5,6,7,8,9], names=head)

head2=['mbase','DeltaA',  'FWHM',  'devi', 'Tmax', 'fpl'  , 'rho' , 'tstar' , 'ur' ,  'fb' , 'mstar', 'limb']
df= pd.read_csv("./paramall2b.txt", sep=" ",  skipinitialspace = True, header=None, usecols=[0,1,2,3,4,5,6,7,8,9, 10, 11], names=head2)

print("describe:  ",  df.describe())
print("Columns:  ",  df.columns, "len(columns):  ",  len(df.columns)   )
print("*****************************************") 
###########################################################################
fij=open("./ntree_2.txt","a")
fij.close()

ntre=([1,2,3,4,5,6,7,8,9,10,13, 15, 17, 20, 25, 30, 37, 45, 55, 70, 80, 90, 100, 120, 140, 170, 200, 240, 280, 320, 380, 450, 500])
for i in range(33):
    i+=28
    Ntree=int(ntre[i])
    forest = RandomForestRegressor(n_estimators=int(Ntree),random_state = 65)
    out1=np.zeros((25))
    out1[0]=int(Ntree)
    for j in range(ns): 
        x=np.zeros(( len(df.tstar) , 6))
        y=np.zeros(( len(df.tstar)    ))
        for k in range(len(df.tstar)):  
            #x[k,0], x[k,1], x[k,2], x[k,3], x[k,4]= df.mbase[k], df.DeltaA[k], df.FWHM[k], df.devi[k], df.Tmax[k]
            x[k,0], x[k,1], x[k,2], x[k,3], x[k,4], x[k,5]= df.mbase[k], df.DeltaA[k], df.FWHM[k], df.devi[k], df.Tmax[k], df.fpl[k]
            if(j==0): y[k]= df.rho[k] 
            if(j==1): y[k]= df.tstar[k]
            if(j==2): y[k]= df.ur[k]
            if(j==3): y[k]= df.fb[k]
            if(j==4): y[k]= df.mstar[k]
            if(j==5): y[k]= df.limb[k]
            
        xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.2, random_state=65)   
        forest.fit(xtrain, ytrain)  
        ypred = forest.predict(xtest)
        mape =np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
        mse= metrics.mean_squared_error(ytest, ypred)  
        r2s= metrics.r2_score(ytest, ypred)
        rmse=np.sqrt(mse)
        #out1.append(mape)
        #out1.append(mse)
        #out1.append(r2s)
        #out1.append(rmse)
        out1[j*4+1], out1[j*4+2], out1[j*4+3], out1[j*4+4]= mape, mse, r2s, rmse
        print("Ntree, output,   score, MAE, MSE, RMSE:    ", Ntree,    j,   mape,    mse,    r2s,  rmse   )
    fij=open("./ntree_2.txt","a")
    np.savetxt(fij,out1.reshape((-1,25)),fmt="%d   %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f  %.10f") 
    fij.close()
    print("*********************************************************")
    
    
    
    
    
    
    
    
    
