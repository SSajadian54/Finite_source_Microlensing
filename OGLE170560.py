import numpy as np 
import pylab as py 
import pandas as pd
import matplotlib.pyplot as plt 

##########################################################################
f3=open("./OB170560/c_0.dat","r")
ndd=int(sum(1 for line in f3) )
mod =np.zeros(( ndd , 4))
mod=np.loadtxt("./OB170560/c_0.dat") 

##########################################################################

f1=open("./OB170560/data_tot.txt","r")
nt=int(sum(1 for line in f1) )
dat0 =np.zeros(( nt , 3))
dat  =np.zeros(( nt , 3))
dat0= np.loadtxt("./OB170560/data_tot.txt") 
col=[]
test= dat0[:,0].argsort()
print(test)
for i in range(nt):  
    dat[i,:]=dat0[int(test[i]),:]
    if(dat[i,2]==0):    col.append('b')
    if(dat[i,2]==1):    col.append('g')
    if(dat[i,2]==2):    col.append('r')
    if(dat[i,2]==3):    col.append('y')

mbase= np.max(dat[:,1])

dat[:,1]=np.power(10.0,-0.4*(dat[:,1]-mbase) ) 


#####################################################   
   
Amin=1.0;## np.min(dat[:,1])
Amax= np.max(dat[:,1])
DeltaA= abs(Amax-Amin)
t0=   dat[np.argmax(dat[:,1]),0]-0.08
print ("Magnification min, max, t0:  ",  Amin,  Amax, t0)
print ("DeltaA:  ",   DeltaA)


#####################################################


epsi=1000.0;  
yHM=0.0
for i in range(nt): 
    dA=abs(dat[i,1]-Amin)
    if(abs(dA-0.5*DeltaA)<epsi):  
        epsi= abs(dA - 0.5*DeltaA); 
        tHM= float(dat[i,0])
        yHM= float(dat[i,1])
FWHM= 2.0*abs(tHM-t0)
print ("Time_H,  FWHM, yHM:    ",  tHM,  2.0*abs(tHM-t0),   yHM)
if(tHM>t0): 
    x1, x2=tHM-FWHM,    tHM
else:
    x1, x2= tHM,  tHM+FWHM  
print ("x1, x2:  ",  x1, x2     ) 

##########################################################################33
maxd=0.0; Tmax=0.0;   cmax=0
fif=open("./OB170560/deri.txt","w")   
deri=np.zeros(nt-1)
for i in range(nt-1):
    if(abs(dat[i+1,0]-dat[i,0])<1.0 and dat[i+1,0]!= dat[i,0]):  
        deri[i]=abs((dat[i+1,1]-dat[i,1])/(dat[i+1,0]-dat[i,0]) )
        if(deri[i]>maxd and (abs(dat[i,0]-x1)<0.5 or abs(dat[i,0]-x2)<0.5) ):
            maxd= deri[i]
            Tmax= dat[i,0] 
            cmax=i
    else:  
        deri[i]=0.0001
print("Tmax,  maxd:  ",  Tmax, maxd)    

if(Tmax>t0): 
    t2=Tmax;  t1=float(Tmax-2.0*(Tmax-t0) )    
else: 
    t1=Tmax;  t2=float(Tmax+2.0*(t0-Tmax) )
print("t1, t2:  ",  t1, t2)

devi=0.0;  Num=0.0; 
hat=np.zeros(nt)
for i in range(nt): 
   if(dat[i,0]<t1):                       hat[i]=Amin;
   elif(dat[i,0]>=t1 and dat[i,0]<t2):    hat[i]=Amax;   
   else:                                  hat[i]=Amin;
   Num+=1.0;
   devi+=float(dat[i,1]-hat[i])*(dat[i,1]-hat[i])/hat[i]/hat[i];
devi=float(devi/Num/1.0)
print ("deviation:  ",   devi)


del0=float(Amax-1.0); 
delm=float(dat[cmax,1]-1.0);   
fpl= 1.0 -delm/del0;


print ("****** mbase, DeltaA,  FWHM,  devi, Tmax,   fpl:  ",  mbase, DeltaA, FWHM, devi,  Tmax-t0 , fpl)



#####################################################
plt.cla()
plt.clf()
fig= plt.figure(figsize=(8,6)) 
ax=fig.add_subplot(111)

plt.scatter(dat[:,0],dat[:,1], c=col,marker='o', s=25.0)
plt.plot(dat[:,0],hat,"k--",  lw=1.7, label=r"$\rm{Top}-\rm{Hat}~\rm{model}$")

plt.plot(mod[:,0]+t0,mod[:,1],"m-",  lw=1.7, label=r"$\rm{predicted}~\rm{model}$")


plt.title(r"$\rm{OGLE}-2017-\rm{BLG}-0560$", fontsize=18)
plt.ylabel(r"$\rm{Magnification}$",  fontsize=18)
plt.xlabel(r"$\rm{time(days)}$", fontsize=18)

plt.xlim([7856, 7863])
#plt.ylim([0.96,2.75])

plt.axvline(x=t2, color="b", ls=":", lw=1.8, label=r"$T_{\rm{max}}$")
ax.vlines(x=t0, ymin=Amin, ymax=Amax,  color="g", ls="--", lw=1.8, label=r"$\Delta A$")
ax.hlines(y=yHM , xmin= x1, xmax= x2,  color="r", ls="-",  lw=1.8,  label= r"$\rm{FWHM}$")

plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)

plt.legend()
plt.legend(loc=2,fancybox=True, shadow=True)##, framealpha=0.5)
plt.legend(prop={"size":15})

fig.tight_layout()
plt.savefig("./OB170560/plot.jpg", dpi=200)
print("The magnification is plotted ")
#####################################################
'''
plt.cla()
plt.clf()
plt.scatter(dat[:(nt-1),0],deri ,c=dat[:(nt-1),2], marker='o', s=10.0)
plt.axvline(x=Tmax, color="g", ls="--", lw=1.3)
plt.savefig("./OB170560/Deri.jpg", dpi=200)    
'''  







