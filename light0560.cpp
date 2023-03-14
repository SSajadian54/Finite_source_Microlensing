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

const double Delu= 14.2364536540764615474596; 
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
   fil2=fopen("param0560.txt","w");
   char filnam[40]; 
   VBBinaryLensing vbb;
   vbb.Tol=1.e-6;
   vbb.LoadESPLTable("./ESPL.tbl");

   
  
   double  rho, u, du, umax,max1, deri; 
   int nu=9679;
   double Astar[nu];
   double  u0, fb, pt, u1, tE, mbase;     

   for(int ir=0;  ir<1;  ++ir){

   
   vbb.a1=0.47;
   
   //df.rho[i],   df.tstar[i],  df.ur[i],   df.fb[i],  df.mstar[i] 
   
   rho=0.79;
   tE= 1.18; 
   u0= 0.28;   
   fb=1.0;
   mbase= 14.924; 
   du=double(Delu*rho/nu);   
   
   
   sprintf(filnam,"./files/%c%c%d.dat",'c','_', ir);
   fil1=fopen(filnam,"w");
   
   max1=0.0;        
   umax=0.0;
   for(int i=0;i<nu;++i) Astar[i]=0.0; 
   for(int iu=0;  iu<nu;  ++iu){
   pt= double(-Delu*rho/2.0 + iu*du); 
   u= sqrt(pt*pt+ u0*u0); 
   if(pt<0.0)   u1=-u;
   else         u1=u;
   Astar[iu]=vbb.ESPLMag2(u , rho); 
   if(iu>0)    deri=fabs(Astar[iu]-Astar[iu-1])/du;
   if(iu==0)   deri=0.0;  
   if(deri>max1 and  fabs(u-rho)<fabs(0.5*rho)){max1=deri;  umax=u; }
   fprintf(fil1,"%.8lf     %.8lf    %.8lf   %.7lf\n",pt, fb*Astar[iu]+1.0-fb, deri, u1/rho);}
   fclose(fil1);   
   
   fil2=fopen("param0560.txt","a+");
   fprintf(fil2,"%d  %.10lf  %.10lf  %.10lf  %.10lf  %.10lf  %.6lf\n",ir, rho, u0/rho, fb, umax/rho, sqrt(umax*umax-u0*u0)/rho, vbb.a1); 
   fclose(fil2); 
   cout<<"rho:  "<<rho<<"\t u0/rho:  "<<u0/rho<<"\t blending "<<fb<<endl;
   cout<<"umax:  "<<umax<<"\t  de_max: "<<max1<<endl; }
  

   printf("END_TIME >>>>>>>> %s",ctime(&_timeNow));
   return(0); 
}




















