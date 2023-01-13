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
from matplotlib.ticker import StrMethodFormatter
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model,ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import export_text
from sklearn.metrics import r2_score
from sklearn import tree
cmap=plt.get_cmap('viridis')

f1=open("./ntree_2.txt","r")
nm= sum(1 for line in f1) 
fore=np.zeros(( 27, 25)) 
fore= np.loadtxt("./ntree_2.txt")
#######################################################################
plt.clf()
fig= plt.figure(figsize=(8,6)) 
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,3]), 'b--',lw=2.5, label=r"$\rho_{\star}$")
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,7]), 'm--',lw=2.5, label=r"$t_{\star}$"   )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,11]), 'g--',lw=2.5, label=r"$u_{\rm{r}}$"  )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,15]),'r--',lw=2.5, label=r"$f_{\rm{b}}$"  )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,19]),'k--',lw=2.5, label=r"$m_{\star}$"   )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,23]),'c--',lw=2.5, label=r"$\Gamma$"   )
plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18, rotation=0)
plt.xlabel(r"$\log_{10}[\rm{Number}~\rm{of}~\rm{Trees}]$",fontsize=18)
plt.ylabel(r"$\log_{10}[R2-\rm{score}]$",fontsize=18)
plt.xlim([0.0,2.5])
plt.ylim([-0.3,0.0])
plt.legend()
plt.legend(loc='best',fancybox=True, shadow=True)
plt.legend(prop={"size":18}, loc=4)
plt.grid(True)
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.tight_layout()
fig.savefig("notree2b.jpg", dpi=200)


#######################################################################


f1=open("./ntree_1.txt","r")
nm= sum(1 for line in f1) 
fore=np.zeros(( 33, 21)) 
fore= np.loadtxt("./ntree_1.txt")

#######################################################################
plt.clf()
fig= plt.figure(figsize=(8,6)) 
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,3]), 'b--',lw=2.5, label=r"$\rho_{\star}$")
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,7]), 'm--',lw=2.5, label=r"$t_{\star}$"   )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,11]), 'g--',lw=2.5, label=r"$u_{\rm{r}}$"  )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,15]),'r--',lw=2.5, label=r"$f_{\rm{b}}$"  )
plt.plot(np.log10(fore[:,0]),np.log10(fore[:,19]),'k--',lw=2.5, label=r"$m_{\star}$"   )
plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18, rotation=0)
plt.xlabel(r"$\log_{10}[\rm{Number}~\rm{of}~\rm{Trees}]$",fontsize=18)
plt.ylabel(r"$\log_{10}[R2-\rm{score}]$",fontsize=18)
#plt.yscale('log')
#plt.xscale('log')
plt.xlim([0.0,2.5])
plt.ylim([-0.2,0.0])
plt.legend()
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.grid(linestyle='dashed')
plt.legend(prop={"size":18}, loc=4)
fig=plt.gcf()
fig.tight_layout()
fig.savefig("notree1b.jpg", dpi=200)


