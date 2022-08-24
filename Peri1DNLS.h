#pragma once
#include <stdlib.h>
#include <numeric>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include "Vec_Mat.h"
#include "Fun_Lib.h"
#include "Comp_Geo.h"
using namespace std;


Mat<double> GenBMat();
Mat<double> GenAmat(Point<1> xi_vec, double Delta);
double Wfunc(double ksi, double Delta);
Vec<double> InvoGFunc(Point<1> xi_vec, double Delta, Mat<double> amat);
Vec<double> FCalc(Vec<double> DelU,
	int SysSize, int Mpx,
	vector<vector<int>> NumFam,
	vector<Point<1>> Coords,
	vector<vector<Point<1>>> XiVecFam,
	double Delta, Vec<double> SOL,
	double Delt, double Delv);
void PeriDerGen(Vec<double> SOL,
	int SysSize, int Mpx,
	vector<vector<int>> NumFam,
	vector<Point<1>> Coords,
	vector<vector<Point<1>>> XiVecFam,
	double Delta, double Delv, int tstpc);
int ValIndxFndrE(vector<int> vec, int val);


typedef struct Fiel_Inf {
	Mat<double> Gfuns;
	double Delt;
	double Delv;
	int Pint;
	Vec<double> SOL;
	Vec<double> SOLB;
} F_inf;


class Peri1DNLS {
private:
	double L;								// Length of the 1D Domain.
	double Delta, DeltaC;					// Horizon Radius Coeff and Horizon Distance. (Delta = DeltaC * Delx)
	double Delx;							// Distances Between Material Points in x Dir.
	double Delv;							// Infinitesimal Volume for Material Points.
	int Mpx;								// Number of Material Points in x Dir.
	int Smax;								// Maximum Number of Family Points Allowed.
	double Delt;							// Time Increment;
	double TTM;								// Terminal Time;
	int TSN;								// Number of Time Steps;
	pfn IntCond;							// Initial Condition.
	int NNZCoeff;							// Number of Nonzero Element Contribution to Quasi-Newton Iteration Matrix B.
	int SysSize;							// Size of the Global System.
	vector<int> BDIndx;						// Boundary Material Point Indices.
	vector<Point<1>> Coords;				// Vector for Material Point Coordinates.
	vector<vector<int>> NumFam;				// Vector of Vectors for Family Member Indices.
	vector<vector<Point<1>>> XiVecFam;		// Vector of Vectors for Family Member Indices.
	SMat BMat;								// Sparse Quasi-Newton Iteration Matrix.
	double Eps;								// Error Tolerence for Broyden's Method.
	int Maxit;								// Maximum Iteratrions for Broyden's Method.
	Vec<double> SOL;						// Vector for Solution of the System.
	Vec<double> INCSOL;						// Vector for Incremental Step of Solution.
	int NNZ;								// Number of Non-Zeros in Global Sytem.
	const char *Fn;							// Filename for Output File.
	const char *TPL;						// Tepclot Location in Computer.
	const char *STP;						// System Command char for Printing.
	string Ttl;								// Plot Title.
	string IndVar;							// Name of the Independent Variable.
	string DepVar;							// Name of the Dependent Variable.
	string ZoneN;							// Name of the Zone.
public:
	void Peri1DMesh(double Length, int Matpt_x, int MaxFM,
		double HorizonRadCoeff);			// Initializer for Discretization Class.
	void Peri1DSysInt();					// System Construction.
	void Gensys(double DT, double TT,
		pfn InC);							// System Generator.
	friend Mat<double> GenBMat();					// B Matrix Generator.
	friend Mat<double> GenAmat(Point<1> xi_vec,
		double delta);						// A Matrix Generator.
	friend Vec<double> InvoGFunc(Point<1> xi_vec,
		double Delta, Mat<double> amat);			// G Function Invoker.
	friend double Wfunc(double ksi,
		double Delta);						// Weight Function.
	friend Vec<double> FCalc(Vec<double> DelU,
		int SysSize, int Mpx,
		vector<vector<int>> NumFam,
		vector<Point<1>> Coords,
		vector<vector<Point<1>>> XiVecFam,
		double Delta, Vec<double> SOL,
		double Delt, double Delv);
	friend void JacobCalc(SMat& BMat, Vec<double> SOL,
		int SysSize, int Mpx,
		vector<vector<int>> NumFam,
		vector<Point<1>> Coords,
		vector<vector<Point<1>>> XiVecFam,
		double Delta, double Delt,
		double Delv, Vec<double> DelU);
	void Peri1DSolve(double Tolerence,
		int MaximumIt);						// Solving the System.
	friend void PeriDerGen(Vec<double> SOL,
		int SysSize, int Mpx,
		vector<vector<int>> NumFam,
		vector<Point<1>> Coords,
		vector<vector<Point<1>>> XiVecFam,
		double Delta, double Delv, int tstpc);
};