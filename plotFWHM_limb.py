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
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
#########################################################################################  
def func(x, c1, c2, c3, c4):
    #return(c1*np.exp(-0.29*(x+c2)**c3)+0.99)
    return(np.exp(c1 + c2*x + c3*(x**2.0))+c4)
    #return(c1+ c2*(x-c3)**0.5 + c4*x**-1.0)
#########################################################################################   
n0=10   
nd=9759
ufw=np.zeros((nd,n0+1))
ufw=np.loadtxt("./FWHM_new2.txt") 
#########################################################################################
fill=open('./c123_u0Limb.txt',"w"); 
fill.close()
rho= np.linspace(0.00005,9.8, 1000)
plt.clf()
plt.figure(figsize=(8,6))
for i in range(n0): 
    i+=1
    ini1 = np.array([1.0, 0.5, 0.3, 0.8])
    fitt, pcov = curve_fit(func, ufw[:,0], ufw[:,i], ini1, maxfev = 200000)
    print "******************************************************************"
    print "c1:  ",   fitt[0] ,  "+/-:  ",  np.sqrt(pcov[0,0]),  1.9/(i*0.1+1)
    print "c2:  ",   fitt[1] ,  "+/-:  ",  np.sqrt(pcov[1,1]),  2.5/(i*0.1+1)
    print "c3:  ",   fitt[2] ,  "+/-:  ",  np.sqrt(pcov[2,2]),  0.85/(i*0.1+1)
    print "c4:  ",   fitt[3] ,  "+/-:  ",  np.sqrt(pcov[3,3]),  0.85/(i*0.1+1)
    print "Chi2_Fitt_1: ",  np.sum(np.power( func(ufw[:,0] , *fitt)-ufw[:,1],2.0)*np.power(0.01,-2.0) )
    print "**********************************************************************"
    u0=np.array([(i-1)/20.0, fitt[0], fitt[1], fitt[2], fitt[3]])
    fill=open('./c123_u0Limb.txt',"a")
    np.savetxt(fill,u0.reshape(-1, 5),fmt="$%.2f$ & $%.3f$ & $%.3f$ & $%.3f$ & $%.3f$\\")
    fill.close() 
    plt.scatter(ufw[:,0], ufw[:,i], c="k", s=7.0)   
    plt.plot(rho,func(rho,*fitt),"m--", lw=1.4)
    #plt.plot(rho,func(rho,*ini1),"g--", lw=1.4)
fig=plt.gcf()
fig.savefig("./ufwrho_tot_fitLimb.jpg",dpi=200)
#input("Enter a number")
#########################################################################################

dd= str(r"$\rho_{\star}$")
dd2= str(r"$\rho_{\star}^{2}$")
plt.clf()
plt.figure(figsize=(8,6))
plt.scatter(ufw[:,0], ufw[:,1], c="k", s=10.0)   
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
fig.savefig("./UfwRho_firstLimb.jpg",dpi=200)
#########################################################################################
dd= str(r"$\rho_{\star}$")
dd2= str(r"$\rho_{\star}^{2}$")
plt.clf()
plt.figure(figsize=(8,6))
plt.plot(ufw[:,0], ufw[:,1], "k--", lw=1.45, label=r"$u_0=0$")   
plt.plot(ufw[:,0], ufw[:,2], c="pink",ls="--", lw=1.45, label=r"$u_0=0.1\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,3], "b-.", lw=1.45, label=r"$u_0=0.2\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,4], "m:", lw=1.45, label=r"$u_0=0.3\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,5], "c-", lw=1.45, label=r"$u_0=0.4\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,6], "y--", lw=1.45, label=r"$u_0=0.5\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,7], c="g", ls="-.", lw=1.45, label=r"$u_0=0.6\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,8], c="lime",ls=":", lw=1.45, label=r"$u_0=0.7\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,9], c="orange",ls="-", lw=1.45, label=r"$u_0=0.8\rho_{\star}$") 
plt.plot(ufw[:,0], ufw[:,10],c="r",ls="--", lw=1.45, label=r"$u_0=0.9\rho_{\star}$") 
#plt.plot(rho,func(rho,*ini1),c="k", lw=1.4,label=r"$\rm{Exp}\Big(-1.9 -2.5~$"+str(dd)+r"$-0.85~$"+str(dd2) +r"$\Big)+1.0$")
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.text(7.0,1.05,r"$\Gamma=0.5$",fontsize=18)
plt.xlim([-0.2, 9.9])
plt.ylim([0.92,1.45]) 
plt.xlabel(r"$\rho_{\star}$",fontsize=17)
plt.ylabel(r"$u_{\rm{HM}}/ \rho_{\star}$", fontsize=17)
plt.legend()
plt.legend(loc=1,fancybox=False, shadow=False,framealpha=0.01, fontsize=17)
plt.tight_layout()
fig=plt.gcf()
fig.savefig("./fwhmfLimb.jpg",dpi=200)
#########################################################################################
umax=np.zeros((nd,2))
umax=np.loadtxt("./Tmax_new2.txt")  
plt.clf()
plt.figure(figsize=(8,6))
plt.scatter(umax[:,0], umax[:,1]-umax[:,0], c="k", s=20.0)             
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.ylim([-0.0001,0.0001])
plt.xlabel(r"$\rho_{\star}$",fontsize=18)
plt.ylabel(r"$u_{\rm{max}}-\rho_{\star}$",fontsize=18)
plt.grid("True")
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.savefig("./UmaxRhoLimb.jpg",dpi=200)

#########################################################################################
'''
plt.clf()
plt.figure(figsize=(8,8))
fig, ax= plt.subplots()
im=plt.imshow(np.log(ufw[:,1:]) ,cmap=cmap,extent =[0.0,1.0,ufw[0,0], ufw[nd-1,0]],interpolation ='nearest', origin ='lower',aspect='auto') 
cb=plt.colorbar(im,ax=ax)
cb.ax.tick_params(labelsize=12)
cb.set_label(r"$u_{\rm{HM}}/\rho_{\star}$", fontsize=17)  
plt.ylim([9.7,0.0]) 
plt.xlim([0.0,1.0])             
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.ylabel(r"$\rho_{\star}$",fontsize=18)
plt.xlabel(r"$u_{0}(\rho_{\star})$", fontsize=18)
#plt.grid("True")
#plt.grid(linestyle='dashed')
plt.tight_layout()
fig=plt.gcf()
fig.savefig("./imshow.jpg",dpi=200)
'''
#############################################################
'''
cc=np.zeros((n0))
for i in range(n0):
    cc[i]=float(i+1.0)/n0

#cmap = list(reversed(sns.color_palette("viridis", n0).as_hex() ) )
cmap1 =mpl.cm.jet## mpl.cm.get_cmap('viridis', n0)
norm = mpl.colors.Normalize(vmin=0.0, vmax=0.9)
cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=cmap1)
cmap2.set_array([])
   
plt.clf()
plt.figure(figsize=(8,6))
fig, ax= plt.subplots()
for i in range(n0): 
    ax.scatter(ufw[:,0],ufw[:,i+1], c=cmap2.to_rgba(i*0.1), s=1.2)
cb=fig.colorbar(cmap2, ticks=cc)
cb.ax.tick_params(labelsize=16)
cb.set_label(r"$u_{0}/\rho_{\star}$", fontsize=17)                
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
plt.xlim([-0.2,9.7]) 
plt.ylim([0.98,1.5]) 
plt.xlabel(r"$\rho_{\star}$",fontsize=17)
plt.ylabel(r"$u_{\rm{HM}}/ \rho_{\star}$", fontsize=17)
plt.grid("True")
plt.grid(linestyle='dashed')
plt.tight_layout()
fig=plt.gcf()
fig.savefig("./UhwRho.jpg",dpi=200)
'''





