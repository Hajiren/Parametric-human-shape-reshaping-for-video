#ifndef _TESTSO_H  
#define_TESTSO_H  

extern"C"  
{ 
int insertConstraint(int x1,int y1,int x2,int y2); 
int calculateCDT(int,int);
int clearCDT();
int insertPoint(int x1,int y1);
typedef int insertPoint(int,int);
typedef int insertConstraint(int,int,int,int);
typedef int calculateCDT(int,int);
typedef int clearCDT();
} 

#endif 