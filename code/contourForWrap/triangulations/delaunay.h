#ifndef _TESTSO_H  
#define_TESTSO_H  

extern"C"  
{ 
int insertConstraint(int x1,int y1,int x2,int y2); 
int calculateCDT();
typedef int insertConstraint(int,int,int,int);
typedef int calculateCDT();
} 

#endif 