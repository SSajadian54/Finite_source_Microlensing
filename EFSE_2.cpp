#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include "VBBinaryLensingLibrary.h"
using namespace std;
///===============================================
    time_t _timeNow;
    unsigned int _randVal;
    unsigned int _dummyVal;
    FILE * _randStream;
//===============================================
int main(){
//================================================
   time(&_timeNow);
   _randStream = fopen("/dev/urandom", "r");
   _dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
   srand(_randVal);
   _dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
   srand(_randVal);
///=================================================
   printf("START_TIME >>>>>>>> %s",ctime(&_timeNow));
   FILE*  fil1;
   FILE* fil2;   
   fil2=fopen("./paramall2.txt","w");

   VBBinaryLensing vbb;
   vbb.Tol=1.e-5;
   vbb.LoadESPLTable("./ESPL.tbl");


   char filnam[40]; 
   int N, Num, imax; 
   double minm, maxm, dt; 
   double pt1, pt2, tmax, tfw;  
   double tstar, u,  tim, Astar0, fact, limb;
   double u0,    tE,   rho,  fb,  mstar; 
   double mbase, DeltaA, FWHM, devi, Tmax;
   double del0, delm, fpl; 

 
   for(int icon=0; icon<700000; ++icon){
  
   do{
   rho=double((double)rand()/(double)(RAND_MAX+1.))*9.0+1.0;
   u0=double((double)rand()/(double)(RAND_MAX+1.))*10.0; 
   }while(u0>fabs(rho));
   tE=double((double)rand()/(double)(RAND_MAX+1.))*4.0+3.0;
   fb=double((double)rand()/(double)(RAND_MAX+1.))*1.0;
   mstar=double((double)rand()/(double)(RAND_MAX+1.))*4.0+16.0;
   vbb.a1=double((double)rand()/(double)(RAND_MAX+1.))*0.35+0.35;
   limb=vbb.a1; 
   
   
   mbase= double(mstar+2.5*log10(fb));//[1]
   tstar= double(rho*tE); 
   dt=double(tstar/9852.623452); 
   pt1= 0.0;
   pt2= 2.52566666663415*tstar; 
   N= int((pt2-pt1)/dt)+1;
   cout<<"N:  "<<N<<endl;
   double Astar[N], deri[N], Hat[N];
  
  
  
   maxm=0.0;  tmax=0.0;  imax=0; 
   for(int i=0; i<N; ++i){
   tim=double(pt1 + i*dt);
   u=sqrt(u0*u0 + tim*tim/tE/tE);
   Astar[i]=vbb.ESPLMag2(u , rho)*fb+1.0-fb;
   if(i>0) {fact=tE*u/sqrt(u*u-u0*u0);  deri[i]=fabs(Astar[i]-Astar[i-1])/dt*fact;}
   else     deri[0]=0.0;
   if(deri[i]>maxm and fabs(u-rho)<double(0.2*rho) and fabs(log10(deri[i]/deri[i-1]))<2.5 and i>0){maxm= deri[i];   tmax=tim;   imax=i;}}
   
   DeltaA= fabs(Astar[0] - Astar[N-1]);//[2] 
   
   ///*****************************************************************************************
   
   minm=100000.0;  tfw=0.0; 
   Astar0= fabs(Astar[0]-Astar[N-1])*0.5;
   Num=0.0;  devi=0.0; 
   for(int i=0; i<N; ++i){ 
   tim=double(pt1 + i*dt);
   u=sqrt(u0*u0 + tim*tim/tE/tE);
   if(i<imax)        Hat[i]=Astar[0];
   else if(i==imax)  Hat[i]=Astar[i];   
   else if(i>imax)   Hat[i]=Astar[N-1];
   Num+=1.0;
   devi+=double(Astar[i]-Hat[i])*(Astar[i]-Hat[i])/Hat[i]/Hat[i];
   if(fabs(fabs(Astar[i]-Astar[N-1])-Astar0)<minm){ minm=fabs(fabs(Astar[i]-Astar[N-1])-Astar0); tfw=tim;}}

   FWHM=tfw*2.0;//2.0*tE*sqrt(ufw*ufw-u0*u0); //days[3]
   devi=double(devi/Num);//[4]
   Tmax=tmax;  //tE*sqrt(umax*umax-u0*u0);//days[5]
   
   del0=double(Astar[0]-1.0); 
   delm=double(Astar[imax]-1.0);   
   fpl= 1.0 -delm/del0;
   
      
   ///*****************************************************************************************
   if(icon%10000==0){
   sprintf(filnam,"./files/%c%c%d.dat",'k','_',icon);
   fil1=fopen(filnam,"w");
   for(int j=0; j<N; ++j){
   tim= double(pt1 + j*dt); 
   fprintf(fil1,"%.5lf    %.10lf     %e   %.7lf\n",tim/tstar, Astar[j], deri[j], Hat[j]);}
   fclose(fil1);}


   if(FWHM<0.0 or FWHM==0.0 or Tmax==0.0 or fabs(sqrt(Tmax*Tmax/tE/tE+u0*u0)-rho)>double(0.8*rho) or devi>5.0 or Astar0<0.0){
   cout<<"Error!!! icon:      "<<icon<<"\t u0/rho:  "<<u0/rho<<"\t fb:  "<<fb<<endl;
   cout<<"FWHM:  "<<FWHM<<"\t Tmax:  "<<Tmax<<"\t tmax:  "<<tmax<<endl;
   cout<<"devi:  "<<devi<<"\t Astar0:  "<<Astar0<<endl;
   int eew; cin>>eew;}
   
   
   
   fprintf(fil2,"%.8lf  %.8lf  %.8lf  %.8lf  %.8lf  %.8lf  %.8lf  %.8lf  %.8lf  %.8lf  %.8lf   %.8lf\n",
   mbase,DeltaA,FWHM,devi,Tmax,fpl, rho,tstar,u0/rho,fb,mstar,limb);
   
   cout<<"================================================"<<endl;  
   cout<<"*************>>>   "<<icon<<"  <<< ***************"<<endl; 
   cout<<"Observi mb:  "<<mbase<<"\tDeltaA:  "<< DeltaA<<"\t FWHM: "<<FWHM<<"\tdevi  "<<devi<<"\tTmax: "<<Tmax<<"\t f_pl:  "<<fpl<<endl;
   cout<<"Lensing tE:  "<<tE<<"\trho:  "<<rho<<"\tu0:  "<<u0<<"\tfb: "<<fb<<"\tmag*:  "<<mstar<<"\t limb:  "<<limb<<endl;
   cout<<"================================================"<<endl;}
   fclose(fil2); 
   printf("END_TIME >>>>>>>> %s",ctime(&_timeNow));
   return(0); 
}
