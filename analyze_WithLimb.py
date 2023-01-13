import time
import numpy as np 
import pylab as py 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from matplotlib import rcParams
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
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
cmap=plt.get_cmap('viridis')

head=[ 'mbase' , 'DeltaA' ,  'FWHM',  'devi', 'Tmax', 'fpl'  , 'rho' , 'tstar' , 'ur' ,  'fb' , 'mstar', 'limb']
labell=[r"$m_{\rm{base}}$",r"$\Delta A$",r"$\rm{FWHM}$", r"$\Delta_{\rm{TH}}$",r"$T_{\rm{max}}$", r"$f_{\rm{pl}}$",r"$\rho_{\star}$",r"$t_{\star}$", r"$u_{\rm{r}}$",r"$f_{\rm{b}}$",r"$m_{\star}$", r"$\Gamma$"]
df= pd.read_csv("./paramall2.txt", sep=" ",  skipinitialspace = True, header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11], names=head)
print("describe:     ",  df.describe())
print("Columns:  ",  df.columns, "len(columns):  ",  len(df.columns)   )
print("******************************************************")
#######################################################################

corrM = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
corrM.style.background_gradient(cmap='coolwarm').set_precision(2)
ax= sns.heatmap(corrM, annot=True, xticklabels=labell, yticklabels=labell,annot_kws={"size": 16}, square=True, linewidth=1.0, cbar_kws={"shrink": .99}, linecolor="k",fmt=".2f", cbar=True, vmax=1, vmin=-1, center=0.0, ax=None, robust=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.xticks(rotation=45,horizontalalignment='right',fontweight='light', fontsize=18)
plt.yticks(rotation=0, horizontalalignment='right',fontweight='light', fontsize=18)
plt.title(r"$\rm{Correlation}~\rm{Matrix}$", fontsize=19)
fig.tight_layout()
plt.savefig("corr2.jpg", dpi=200)
print("**** Correlation matrix was calculated ******** ")

###########################################################################
fif=open("./result2c.txt","a")
fif.close()
f1=open("./paramall2.txt","r")
nm= sum(1 for line in f1) 
par=np.zeros((nm,12)) 
par= np.loadtxt("./paramall2.txt")
##########################################################################
x=np.zeros(( len(df.tstar) , 6))
y=np.zeros(( len(df.tstar) , 6))
for i in range(len(df.tstar)):  
    x[i,0], x[i,1], x[i,2], x[i,3], x[i,4],x[i,5]= df.mbase[i], df.DeltaA[i], df.FWHM[i], df.devi[i], df.Tmax[i], df.fpl[i]
    y[i,0], y[i,1], y[i,2], y[i,3], y[i,4],y[i,5]= df.rho[i],   df.tstar[i],  df.ur[i],   df.fb[i],  df.mstar[i], df.limb[i] 
###########################################################################
for i in range(6): 
    i+=5
    test2=np.zeros((10,4))
    model = RandomForestRegressor(n_estimators=120, n_jobs=1, random_state=65)
    cv = model_selection.KFold(n_splits=10)
    nkf=0; 
    for traini, testi in cv.split(x):
        xtrain=np.zeros((len(traini),6))    
        ytrain=np.zeros((len(traini) ))
        xtest= np.zeros((len(testi), 6))      
        ytest= np.zeros((len(testi)  ))
        for j in range(len(traini)):
            k=int(traini[j]) 
            xtrain[j,:] = x[k , :] 
            ytrain[j] =   y[k , i]
        for j in range(len(testi)):
            k=int(testi[j])
            xtest[j,:] = x[k , :]
            ytest[j] =   y[k , i]
         
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        mape= np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
        mse= metrics.mean_squared_error(ytest, ypred)
        r2s=   metrics.r2_score(ytest, ypred)
        rmse=np.sqrt(mse)

        test2[nkf,0], test2[nkf,1], test2[nkf,2], test2[nkf,3]  = r2s,  mape,   mse,   rmse 
        print("r2s,  mape,   mse,  rmse:    ",  r2s,  mape,   mse,   rmse)
        nkf+=1
    ave=np.array([np.mean(test2[:,0]), np.mean(test2[:,1]),   np.mean(test2[:,2]),  np.mean(test2[:,3]) ])
    print(ave)
    fif=open("./result2c.txt","a+")
    np.savetxt(fif,ave.reshape((-1,4)),fmt="Target_i & $%.3f$ &   $%.3f$  &   $%.3f$  &  $%.3f$ ") 
    fif.close();
print("importance:  ", model.feature_importances_ )
###################################################################  
test2=np.zeros((10,4))
model = RandomForestRegressor(n_estimators=120, n_jobs=1, random_state=65)
cv = model_selection.KFold(n_splits=10)
nkf=0; 
for traini, testi in cv.split(x):
    xtrain=np.zeros((len(traini),6))    
    ytrain=np.zeros((len(traini),6))
    xtest= np.zeros((len(testi), 6))      
    ytest= np.zeros((len(testi), 6))
    for j in range(len(traini)):
        xtrain[j,:] = x[int(traini[j]),:] 
        ytrain[j,:] = y[int(traini[j]),:]
    for j in range(len(testi)):
        xtest[j,:] = x[int(testi[j]),:]
        ytest[j,:] = y[int(testi[j]),:]
         
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    mape= np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
    mse= metrics.mean_squared_error(ytest, ypred)
    r2s=   metrics.r2_score(ytest, ypred)
    rmse=np.sqrt(mse)
    test2[nkf,0], test2[nkf,1], test2[nkf,2], test2[nkf,3]  = r2s,  mape,   mse,   rmse
    print("r2s,  mape,   mse,   rmse:     ",   mape,    mse,    r2s,    rmse)
    nkf+=1
ave=np.array([np.mean(test2[:,0]), np.mean(test2[:,1]),   np.mean(test2[:,2]),  np.mean(test2[:,3]) ])
print (ave)
fif=open("./result2c.txt","a+")
np.savetxt(fif,ave.reshape((-1,4)),fmt="AVE_TOT &  $%.3f$ &   $%.3f$   &  $%.3f$   &  $%.3f$ ") 
print("importance:  ", model.feature_importances_)
np.savetxt(fif,model.feature_importances_.reshape((-1,6)),fmt="importance &  $%.3f$ & $%.3f$ &  $%.3f$ &  $%.3f$ & $%.3f$ & $%.3f$") 
print(model.estimators_[0].tree_.max_depth)       
fif.close();
###################################################################   

start_time =time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time
print("Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances = pd.Series(importances, index=labell[:6])
plt.clf()
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_ylabel(r"$\rm{Feature}~\rm{importance}$", fontsize=18)
#ax.set_ylabel(r"$\rm{Mean}~\rm{decrease}~\rm{in}~\rm{impurity}$",fontsize=18)
#plt.ylabel('Importance')
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'dashed')
fig.tight_layout()
fig.savefig("import2a.jpg", dpi=200)
###################################################################      
start_time = time.time()
result = permutation_importance(model, xtest, ytest, n_repeats=10, random_state=42, n_jobs=1)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
plt.clf()
fig, ax = plt.subplots()
forest_importances = pd.Series(result.importances_mean, index=labell[:5] )
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_ylabel(r"$\rm{Feature}~\rm{importance}$", fontsize=18)
#ax.set_ylabel(r"\rm{Mean}~\rm{accuracy}~\rm{decrease}", fontsize=18)
plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'dashed')
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
fig.tight_layout()
fig.savefig("import2b.jpg", dpi=200)
     
#################################################################################
### One decision tree     
tre=DecisionTreeRegressor(max_depth=13) 
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.2, random_state=0)    
tre.fit(xtrain, ytrain)
ypred=tre.predict(xtest);
mape =np.abs(np.mean(np.abs((ytest-ypred)/ytest) ))
mse= metrics.mean_squared_error(ytest, ypred)  
r2s= metrics.r2_score(ytest, ypred)
rmse=np.sqrt(mse)
resu=np.array([r2s,   mape,    mse,    rmse])
fif=open("./result2c.txt","a+")
np.savetxt(fif,resu.reshape((1, 4)),fmt="$\rm{ONE}~\rm{TREE}$ & $%.3f$ &  $%.3f$   &  $%.3f$  &  $%.3f$") 
fif.close();
print("************* One decision tree  **********************") 
print("r2s,   mape,    mse,    rmse:        ", r2s,   mape,    mse,    rmse)
array=np.zeros((len(xtest) , 3))
for i in range(len(xtest)): 
    array[i,0]= float(xtest[i,0])
    array[i,1]= float(ytest[i,4])
    array[i,2]= float(ypred[i,4])
plt.clf()
plt.scatter(array[:,0], array[:,1],c="m", s=3.0, label="original")
plt.scatter(array[:,0], array[:,2],c="b", s=3.5, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel(r"$m_{\rm{base}}$", fontsize=18)
plt.ylabel(r"$m_{\star}$",     fontsize=18)
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.grid(linestyle='dashed')
plt.savefig("Example_mbaseLimb.jpg",  dpi=200) 
##################################################################
#rep=tree.export_text(tre,feature_names= labell[:5])
#with open("decistion_tree.txt", "w") as fout:
#    fout.write(rep)   
#fig = plt.figure(figsize=(25,20))
#_ =tree.plot_tree(tre, filled=True) ## feature_names=labell[:5],
#fig.savefig("DTC"+ ".jpg", format="jpg", dpi=250, bbox_inches='tight') 
nd=int(30)
err=np.zeros((nd,3))
for j in range(nd):   
    err[j,0]=j+1
    tree=DecisionTreeRegressor(max_depth=j+1)   
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2, random_state=0)    
    tree.fit(xtrain, ytrain)
    ypred= tree.predict(xtest)
    err[j,1]= metrics.mean_squared_error(ytrain , tree.predict(xtrain))
    err[j,2]= metrics.mean_squared_error(ytest ,  ypred)
    print ("Error(Training Set):  ",  err[j,1],  "  Error(test set):  ",  err[j,2])
    print ("*********************************************")
plt.clf()
plt.plot(err[:,0], err[:,1], "r-",  lw=2.0, label=r"$\rm{Training}~\rm{Set}$")
plt.plot(err[:,0], err[:,2], "b--", lw=2.0, label=r"$\rm{Test}~\rm{Set}$")
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$\rm{Maximum}~\rm{Tree}~\rm{Depth}$", fontsize=17)
plt.ylabel(r"$\rm{Mean}~\rm{Squared}~\rm{Error}$", fontsize=17)
plt.legend()
plt.savefig("ErrorDepthLimb.jpg", dpi=200)
################################################################################33
### Scatter plots 
plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot=plt.scatter(abs(par[:,7]),abs(par[:,2]),  c=par[:,8],  cmap=cmap, s=1.5 )
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$u_{0}/\rho_{\star}$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$t_{\star}$",fontsize=18)
plt.ylabel(r"$FWHM$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter1L.jpg",dpi=200)
#######################################################
plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot=plt.scatter(abs(par[:,7]),abs(par[:,4]), c=par[:,8], cmap=cmap, s=1.4)
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$ u_{0}/\rho_{\star}$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$t_{\star}$",fontsize=18)
plt.ylabel(r"$T_{max}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter2L.jpg",dpi=200)

#######################################################

plt.clf()
fig= plt.figure(figsize=(8,6))
ax= plt.gca()              
plot= plt.scatter(abs(par[:,2]),abs(par[:,7]),c=par[:,8], cmap=cmap, s=1.4 )
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$\log_{10}[\rho_{\star}]$", fontsize=17)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.ylabel(r"$|FWHM/2 -T_{max}|$",fontsize=18)
plt.xlabel(r"$u_{r}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./lights/scatter3L.jpg",dpi=200)
print("Scatter plots were made *****************************")
###################################################################
for i in range(nm): 
    mbase,  DeltaA,   FWHM,  devi, Tmax, fpl=par[i,0], par[i,1], par[i,2], par[i,3], par[i,4],  par[i,5]
    rho, tstar,  ur,   fb,  mstar, limb=     par[i,6], par[i,7], par[i,8], par[i,9], par[i,10], par[i,11]
    if(i%1000==0):  
        print("observable features:  ", mbase, DeltaA,  FWHM,  devi,  Tmax , fpl)
        print("Lensing parameters:  ",  rho,  tstar, ur,  fb, mstar, limb )
        f2=open("./files/n_{0:d}.dat".format(i),"r")
        nd= sum(1 for line in f2)
        print ("No.  data:  ",  nd)
        if(nd>0):  
            dat=np.zeros((nd,4)) 
            dat=np.loadtxt("./files/n_{0:d}.dat".format(i))
            plt.clf()
            fig= plt.figure(figsize=(8,6)) 
            plt.scatter(dat[:,0], dat[:,1], color= "m", s=1.0)     
            plt.scatter(dat[:,0], dat[:,3], color= "g", s=1.0)  
            plt.axvline(x=FWHM/tstar/2.0,color='red', linestyle='--', lw=1.3)     
            plt.axvline(x=Tmax/tstar,color='pink', linestyle='--', lw=1.3) 
            plt.axhline(y=DeltaA+dat[nd-1,1],color='k', linestyle=':', lw=1.3)                 
            plt.title(r"$\rho_{\star}=$"+str(round(rho,2))+r"$,~~t_{\star}=$"+str(round(tstar,2))+r"$,~~u_{r}=$"+
               str(round(ur,2))+r"$,~~f_{b}=$"+str(round(fb,2)),fontsize=18,color="k")
            plt.xlabel(r"$time(t_{\star})$",fontsize=18,labelpad=0.1)
            plt.ylabel(r"$Magnification$",  fontsize=18,labelpad=0.1)
            plt.xticks(fontsize=16, rotation=0)
            plt.yticks(fontsize=16, rotation=0)
            py.xlim([ 0.0 , 6.0 ])
            fig=plt.gcf()
            fig.savefig("./lights/lightB_{0:d}.jpg".format(i),dpi=200)
            ###########################################################3
            plt.clf()
            fig= plt.figure(figsize=(8,6)) 
            plt.scatter(dat[:,0], dat[:,2], color="m", s=1.0)       
            plt.xlabel(r"$time(t_{\star})$",fontsize=18,labelpad=0.1)
            plt.ylabel(r"$Derivative$",fontsize=18,labelpad=0.1)
            plt.axvline(x=Tmax/tstar,color='pink', linestyle='--', lw=1.3) 
            plt.xticks(fontsize=16, rotation=0)
            plt.yticks(fontsize=16, rotation=0)
            py.xlim([0.0 , 6.0])
            plt.yscale('log')
            fig=plt.gcf()
            fig.savefig("./lights/derivativeB_{0:d}.jpg".format(i),dpi=200)            
            print("Lightcurves are plotted ************************",   i)
