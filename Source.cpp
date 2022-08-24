#include "Peri1DNLS.h"
#include "adept_source.h"

const double pi = 3.141592654;

double IntCondFunc(double coord_x) {
	//return exp(- 0.5 * (coord_x - 10) * (coord_x - 10));
	return 2.0 / cosh(coord_x);
}


int main() {
	string Fname = "NLS.dat";
	string TecPLoc = "\"D:\\Tecplot360\\Tecplot 360 EX 2016 R2\\bin\\tec360.exe\"";
	string Title = "SOLUTION OF 1D NLS EQUATION";
	string IndVariable;
	IndVariable = "x coordinate";
	string DepVariable = "Disp";
	string Zonename = "PD_Solution";
	Peri1DNLS B;
	B.Peri1DMesh(10.0, 256, 50, 3.015);
	B.Peri1DSysInt();
	B.Gensys(5.0e-4, pi / 2, IntCondFunc);
	B.Peri1DSolve(1e-6, 50);
	system("pause");
	return 0;
}