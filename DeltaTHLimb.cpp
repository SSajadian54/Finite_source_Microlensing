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
//================================================

const int n0=10; 
const double u11= 0.00007815346561473; 
const double Delu= 2.2364536540764615474596; 
const double rho0= 0.0033893653645737564634541; 
const double rhof= 9.797324126542564344855465; 


int main(){

   FILE* fil2;
   FILE* fil1;
   fil2=fopen("Fws_new2.txt","a+");


   char filnam[40]; 
   VBBinaryLensing vbb;
   vbb.Tol=1.e-5;
   vbb.LoadESPLTable("./ESPL.tbl");

   
   double Astar0, rho, u, du, umax, Num, max1; 
   int flagf=0,imax; 
   int count=0; 
   int nr=19834;//9759;
   int nu=96709;
   double Astar[nu];
   double Hat[nu];
   double deri[nu];
   double devi[n0]={0.0};  
   double u0[n0]={0.0}; 
   for(int i=0; i<n0; ++i) u0[i]=double(0.0+(1.0/n0)*i);  
   

  
   for(int ir=0;  ir<nr;  ++ir){
   rho=double(rho0+ ir*rhof/nr);
   du=double(Delu*rho/nu);   
   
   vbb.a1=0.5;
   
   count+=1;
   flagf=0;
   if(count%1000==0 and rho>9.9){
   flagf=1;
   sprintf(filnam,"./files/%c%c%d.dat",'l','_', count);
   fil1=fopen(filnam,"w");}
   
  
   max1=0.0;  umax=0.0; imax=-1;
   for(int i=0;i<nu;++i){ Astar[i]=0.0;   deri[i]=0.0;} 
   for(int iu=0;  iu<nu;  ++iu){
   u=double(u11+iu*du); 
   Astar[iu]=vbb.ESPLMag2(u , rho); 
   if(iu>0)    deri[iu]=fabs(Astar[iu]-Astar[iu-1])/du;
   if(iu==0)   deri[iu]=0.0;  
   if(deri[iu]>max1 and fabs(u-rho)<fabs(0.5*rho)){max1=deri[iu]; umax=u; imax=iu; }} 


  
   fil2=fopen("Fws_new2.txt","a+");
   fprintf(fil2,"%.10lf   %.8lf   %.8lf   ",rho, umax/rho, max1);
   for(int i=0; i<n0; ++i){
    
   devi[i]=0.0;   
   Num=0.0;
   Astar0=-1.0;
   for(int iu=0;  iu<nu;  ++iu){
   Hat[iu]=0.0; 
   u=double(u11+iu*du);  
   if(u>double(u0[i]*rho)){
   if(Astar0<0.0)    Astar0=Astar[iu];  
   if(iu<imax)       Hat[iu]=Astar0;
   else if(iu==imax) Hat[iu]=Astar[iu];   
   else if(iu>imax)  Hat[iu]=Astar[nu-1];
   if(flagf>0 and i==0) fprintf(fil1,"%.5lf   %.5lf    %.5lf    %.5lf  %.5lf\n",rho,u/rho, Astar[iu], deri[iu], Hat[iu]);
   Num+=1.0;
   devi[i]+=double(Astar[iu]-Hat[iu])*(Astar[iu]-Hat[iu])/Hat[iu]/Hat[iu];}}
   
   devi[i]= double(devi[i]/Num);
   fprintf(fil2,"%.8lf  ", devi[i]);}
   
   fprintf(fil2,"\n");
   fclose(fil2);
   if(flagf>0)    fclose(fil1);     
   cout<<"*** rho:  "<<rho<<"\t umax[rho*]:  "<<umax<<"\t max_dervivative:  "<<max1<<"\t imax:  "<<imax<<endl;
   for(int i=0; i<n0; ++i)  cout<<  devi[i]  <<"\t";
   cout<<"\n=======================================\n"<<endl;
   
   if(imax<0 or imax>nu or fabs(umax-rho)>double(rho*0.4) or Astar0>Astar[0] or Astar0<Astar[nu-1] or Num==0.0){
   cout<<"Error imax:  "<<imax<<"\t umax/rho :  "<<umax/rho<<"\t rho:  "<<rho<<endl;
   cout<<"Astar0:  "<<Astar0<<"\t Astar[0]:   "<<Astar[0]<<"\t Astar[nu-1]:  "<<Astar[nu-1]<<"\t num:  "<<Num<<endl;
   //int uue; cin>>uue;
   } 
   
   
   }//end of loop rho

   printf("END_TIME >>>>>>>> %s",ctime(&_timeNow));
   return(0); 
}
