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
const int nl=20;
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



   FILE* fil2;      
   fil2=fopen("./FPL_new.txt","w");
   char filnam[40]; 
   VBBinaryLensing vbb;

   vbb.LoadESPLTable("./ESPL.tbl");
   
   
   double rho; 
   int nr=3599;
   int nu=2;
   double Astar, del0, delm, u00, u01;
   double fpl[nl]={0.0}; 
   
   

   for(int ir=0;  ir<nr;  ++ir){
   rho=double(0.00436536457363454146+ ir*9.732412654256434/nr);
   fil2=fopen("./FPL_new.txt","a+");
   fprintf(fil2,"%.10lf  ",rho );
   
   
   
   for(int j=0; j<nl; ++j){
   vbb.a1= double(0.400735634656354+j*0.400005463546235/nl);
   if(vbb.a1<0.5)    vbb.Tol=1.e-6;
   else              vbb.Tol=1.e-5;
   
   for(int iu=0;  iu<nu;  ++iu){   
   //u00=double((double)rand()/(double)(RAND_MAX+1.))*0.00000000365462345645; 
   //u01=double((double)rand()/(double)(RAND_MAX+1.))*0.0000000063459134785647856; 
   if(iu==0) {Astar=vbb.ESPLMag2(0.0000000354,    rho);    del0=double(Astar-1.0); }
   if(iu==1) {Astar=vbb.ESPLMag2(rho+0.00000006354,rho);    delm=double(Astar-1.0); } }
   fpl[j]= 1.0 - delm/del0;
   fprintf(fil2,"%.10lf   ", fpl[j]); 

   if(delm<0.0 or del0<0.0  or delm>del0  or vbb.a1>0.9 or vbb.a1<0.3 or rho>10.0 or fpl[j]<0.0){
   cout<<"Error delm:  "<<delm<<"\t del0:  "<<del0<<"\t limb_darken:  "<<vbb.a1<<endl;
   cout<<"Error rho:  "<<rho<<"\t fpl[j]:  "<<fpl[j]<<endl;
   int uue; cin>>uue;} 
   cout<<"limb_darkening:  "<<vbb.a1<<endl;}
   fprintf(fil2,"\n");
   fclose(fil2);  
   
   cout<<"*** rho:  "<<rho<<endl;
   for(int i=0; i<nl; ++i)  cout<<  fpl[i]  <<"\t";
   cout<<"\n=======================================\n"<<endl;}

   printf("END_TIME >>>>>>>> %s",ctime(&_timeNow));
   return(0); 
}
