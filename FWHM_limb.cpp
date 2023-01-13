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


   FILE* fil2;  FILE* fil3;     FILE*  fil1;
   fil2=fopen("FWHM_new2.txt","w");
   fil3=fopen("Tmax_new2.txt","w");


   char filnam[40]; 
   VBBinaryLensing vbb;
   vbb.Tol=1.e-4;
   vbb.LoadESPLTable("./ESPL.tbl");

   
   double Astar0, max1, min1, rho, u, du, umax, ufw[n0], deri; 
   int flagf=0; 
   int count=0; 
   int nr=9759;
   int nu=986709;
   double Astar[nu];
   double u0[n0]={0.0}; 
   for(int i=0; i<n0; ++i) u0[i]=double(0.0+(1.0/n0)*i);  
   
   vbb.a1=0.5;

   for(int ir=0;  ir<nr;  ++ir){
   rho=double(0.0436536457363454146+ ir*9.732412654256434/nr);
   du=double(3.2364536540764615474596*rho/nu);   
   
   
   count+=1;
   flagf=0;
   if(count%5000==0){
   flagf=1;
   sprintf(filnam,"./files/%c%c%d.dat",'l','_', count);
   fil1=fopen(filnam,"w");}
   
   
   
   max1=0.0;        
   umax=0.0;
   for(int i=0;i<nu;++i) Astar[i]=0.0; 
   
   
   for(int iu=0;  iu<nu;  ++iu){
   u=double(0.00007815346561473+iu*du); 
   Astar[iu]=vbb.ESPLMag2(u,rho); 
   if(iu>0)    deri=fabs(Astar[iu]-Astar[iu-1])/du;
   if(iu==0)   deri=0.0;  
   if(deri>max1 and  fabs(u-rho)<fabs(0.5*rho)){max1=deri;  umax=u; }
   if(flagf>0) fprintf(fil1,"%.5lf     %.5lf    %.5lf\n",rho,u,Astar[iu]);}
   if(flagf>0) fclose(fil1);   
   
   fil3=fopen("Tmax_new2.txt","a+");
   fprintf(fil3,"%.10lf    %.10lf\n",rho, umax); 
   fclose(fil3); 
   
   
   fil2=fopen("FWHM_new2.txt","a+");
   fprintf(fil2,"%.10lf   ",rho);
   for(int i=0; i<n0; ++i){ 
   Astar0=-1.0; 
   min1=10000000.0;   
   ufw[i]=0.0;   
   for(int ju=0;  ju<nu;  ++ju){
   u=double(0.00007815346561473+ju*du); 
   if(u > double(u0[i]*rho)){
   if(Astar0<0.0) Astar0=fabs(Astar[ju]-1.0)*0.5;
   if(Astar0>0.0 and fabs(fabs(Astar[ju]-1.0)-Astar0)<min1){min1=fabs(fabs(Astar[ju]-1.0)-Astar0); ufw[i]=u;}}}
   fprintf(fil2,"%.8lf  ",ufw[i]/rho);}
   fprintf(fil2,"\n");
   fclose(fil2);
   
   cout<<"*** rho:  "<<rho<<"\t umax[rho*]:  "<<umax<<endl;
   for(int i=0; i<n0; ++i)  cout<<  ufw[i]  <<"\t";
   cout<<"\n=======================================\n"<<endl;}

   printf("END_TIME >>>>>>>> %s",ctime(&_timeNow));
   return(0); 
}
