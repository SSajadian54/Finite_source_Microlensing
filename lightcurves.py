import numpy as np
import matplotlib.pyplot as plt
import pylab as py
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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import StrMethodFormatter

#############################################################################
KP=3.08568025*pow(10.,19); #// in meter.
G=6.67384*pow(10.,-11.0);#// in [m^3/s^2*kg].
velocity= 3.0*pow(10.0,8.0);#//velosity of light
Msun=1.98892*pow(10.,30);# //in [kg].
Rsun= 6.957*pow(10.0, 8.0); #///solar radius [meter]
Mearth= 5.972*pow(10.0,24.0);#// kg
Mjupiter=1.898*pow(10,27.0);#// kg
AU=1.4960*pow(10.0,11.0);
year=365.2421875;
###############################################################################
plt.clf()
nr=int(11)
f0=open("./FWHM_new.txt","r")
nn=sum(1 for line in f0)
par=np.zeros(( nn , nr ))
par=np.loadtxt("./FWHM_new.txt") 

uhm=np.zeros((10))

#######################################
for i in range(nn):
    #print par[i,: ]
    rho,    uhm[0], uhm[1], uhm[2], uhm[3], uhm[4]=par[i,0], par[i,1], par[i,2], par[i,3], par[i,4], par[i,5]
    uhm[5], uhm[6], uhm[7], uhm[8], uhm[9]        =par[i,6], par[i,7], par[i,8], par[i,9], par[i,10]     
    
    if(i%50==0 and i>0):
        f1=open("./files/l_{0:d}.dat".format(i),"r")
        nd= sum(1 for line in f1)  
        dat=np.zeros((nd,3)); 
        dat= np.loadtxt("./files/l_{0:d}.dat".format(i) )
        if(abs(dat[0,0]-rho)>0.01): 
            print "Error rho:  ",  dat[0,0],   rho
            input("Enter a number  ")
        #############################################################33    
        plt.clf()
        fig=plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(111)
        plt.plot(dat[:,1]/dat[0,0],dat[:,2], 'r--', lw =1.2)
        plt.axvline(x=uhm[0],color='darkgreen', linestyle='--', lw=2.3)
        plt.axvline(x=uhm[4],color='green', linestyle='--', lw=2.3)
        plt.axvline(x=uhm[9],color='lime', linestyle='--', lw=2.3)
        plt.xlabel(r"$u(normalized to RE)$", fontsize=18)
        plt.ylabel(r"$\rm{Magnification}$", fontsize=18)
        plt.xticks(fontsize=17, rotation=0)
        plt.yticks(fontsize=17, rotation=0)
        plt.title(r"$R_{\rm E}\rm{(days)}=$"+str(round(dat[0,0],2)),fontsize=18, color="k")
        ax1.grid("True")
        ax1.grid(linestyle='dashed')
        fig=plt.gcf()
        fig.savefig("./lights/li_{0:d}.jpg".format(i),dpi=200)
        print "Lightcurve was plotted:  ",  i
        ###############################################################33    
        
        








