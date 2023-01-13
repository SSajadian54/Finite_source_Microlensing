import numpy as np
import matplotlib.pyplot as plt
import pylab as py
import matplotlib.font_manager
matplotlib.get_cachedir()
from matplotlib import rcParams
from numpy import ma
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
import math
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import collections
import scipy.stats as stats
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import datetime
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
from matplotlib.ticker import (AutoLocator, AutoMinorLocator)
cmap=plt.get_cmap('viridis')
from random import randint
import seaborn as sns

#################################################  
def func(x, c1, c2, c3, c4):
    #return(c1*np.exp(-0.29*(x+c2)**c3)+0.99)
    return( c1*x**0.5 + c2*x + c3*x**2.0 +c4 )
#################################################   
n0=20
nd=3599
fpl=np.zeros((nd,n0+1))
fpl=np.loadtxt("./FPL_new.txt") 

for i in range(nd): 
    if(fpl[i,1]<fpl[i-1,1] and abs(fpl[i,1]-fpl[i-1,1])>0.005 and i>0):  
        print(i)
        input("Enter a number ")



########################################################
rho= np.linspace(0.05,9.7, 1000)
fill=open('./fpl123_u0.txt',"w"); 
fill.close()
plt.clf()
plt.figure(figsize=(8,6))
for i in range(n0): 
    ini1 = np.array([0.267, -0.067,  0.0015, 0.3557 ])
    fitt, pcov = curve_fit(func, fpl[:,0], fpl[:,i+1], ini1, maxfev = 200000)
    print ("******************************************************************" )
    print ("c1:  ",   fitt[0] ,  "+/-:  ",  np.sqrt(pcov[0,0]),  1.9/(i*0.1+1) )  
    print ("c2:  ",   fitt[1] ,  "+/-:  ",  np.sqrt(pcov[1,1]),  2.5/(i*0.1+1)  )
    print ("c3:  ",   fitt[2] ,  "+/-:  ",  np.sqrt(pcov[2,2]),  0.85/(i*0.1+1) )
    print ("c3:  ",   fitt[3] ,  "+/-:  ",  np.sqrt(pcov[3,3]),  0.85/(i*0.1+1) ) 
    print ("Chi2_Fitt_1: ",  np.sum(np.power( func(fpl[:,0] , *fitt)-fpl[:,1],2.0)*np.power(0.01,-2.0) )  )
    print ("**********************************************************************" )
    u0=np.array([0.4+i*0.4/n0, fitt[0], fitt[1], fitt[2], fitt[3]])
    fill= open('./fpl123_u0.txt',"a")
    np.savetxt(fill,u0.reshape(-1,5),fmt="$%.2f$ & $%.3f$ & $%.3f$ & $%.3f$  & $%.3f$\\")
    fill.close() 
       
    plt.scatter(fpl[:,0], fpl[:,i+1], c="k", s=10.0)   
    plt.plot(rho,func(rho,*fitt),"m--", lw=1.4)
    #plt.plot(rho,func(rho,*ini1),"g--", lw=1.4)
fig=plt.gcf()
fig.savefig("./fplrho_{0:d}.jpg".format(i),dpi=200)
########################################################
'''

plt.clf()
plt.figure(figsize=(8,6))
plt.scatter(fpl[:,0], fpl[:,1], c="k", s=10.0)   
plt.plot(rho,func(rho,*ini1),"m--", lw=1.4,label=r"$\rm{Exp}\Big(-1.9 -2.5~$"+str(dd)+r"$-0.85~$"+str(dd2) +r"$\Big)+1.0$")
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$\rho_{\star}$",fontsize=18)
plt.ylabel(r"$u_{\rm{HM}}/\rho_{\star}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
plt.legend()
plt.legend(loc='best',fancybox=True, shadow=True, fontsize=15)
fig=plt.gcf()
fig.savefig("./UfwRho_0.jpg",dpi=200)
input("Enter a number")
'''

########################################################
#dd= str(r"$\rho_{\star}$")
#dd2= str(r"$\rho_{\star}^{2}$")
#rho= np.linspace(0.004,9.7,1000)
plt.clf()
plt.figure(figsize=(8,6))
plt.plot(fpl[:,0], fpl[:,1], "k:",  lw=1.45, label=r"$\Gamma=0.40$")   
plt.plot(fpl[:,0], fpl[:,3], c="pink",   lw=1.45, label=r"$\Gamma=0.44$") 
plt.plot(fpl[:,0], fpl[:,5], "b-.",    lw=1.45, label=r"$\Gamma=0.48$") 
plt.plot(fpl[:,0], fpl[:,7], "m:",     lw=1.45, label=r"$\Gamma=0.52$") 
plt.plot(fpl[:,0], fpl[:,9], "c-",     lw=1.45, label=r"$\Gamma=0.56$") 
plt.plot(fpl[:,0], fpl[:,11], "y-.",    lw=1.45, label=r"$\Gamma=0.60$") 
plt.plot(fpl[:,0], fpl[:,13], "g:",    lw=1.45, label=r"$\Gamma=0.64$") 
plt.plot(fpl[:,0], fpl[:,15], c="lime",  lw=1.45, label=r"$\Gamma=0.68$") 
plt.plot(fpl[:,0], fpl[:,17], c="orange",lw=1.45, label=r"$\Gamma=0.72$") 
plt.plot(fpl[:,0], fpl[:,20],"r--",    lw=1.45, label=r"$\Gamma=0.78$") 
plt.plot(rho,func(rho,*fitt),c="k", ls="-", lw=1.2, label=r"$0.28 \sqrt{\rho_{\star}} -0.06 \rho_{\star} + 0.001 \rho_{\star}^{2} +0.4 $")
#plt.plot(rho,func(rho,*ini1),c="k", lw=1.4,label=r"$\rm{Exp}\Big(-1.9 -2.5~$"+str(dd)+r"$-0.85~$"+str(dd2) +r"$\Big)+1.0$")
#plt.plot(uws[:,0], uws[:,9], "b:", lw=1.0)     
#plt.plot(rho,func(rho,*ini1),"m--", lw=1.4,label=r"$\rm{Exp}\Big(-1.9 -2.5~$"+str(dd)+r"$-0.85~$"+str(dd2) +r"$\Big)+1.0$")
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlim([-0.1,9.7])
plt.ylim([0.4,0.8]) 
plt.xlabel(r"$\rho_{\star}$",fontsize=17)
plt.ylabel(r"$f_{\rm{pl}}$", fontsize=17)
plt.legend()
plt.legend(loc='best',fancybox=False, shadow=False,framealpha=0.01, fontsize=14)
plt.tight_layout()
fig=plt.gcf()
fig.savefig("./fpl.jpg",dpi=200)
################################################################################











