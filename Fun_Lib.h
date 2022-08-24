#pragma once
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "Vec_Mat.h"
using namespace std;

typedef double(*pfn)(double);
typedef double(*pfn2)(double, double);

int gfact(int num);
double trapz_e(double a, double b, pfn fg, int n);
double simps(double a, double b, pfn fg, int n);
double pol_hor(Vec<double> cf, double x, int n);


static double Sign(double a, double b) {
	return a * fabs(b) / b;
}


template <class T>
double ridders(T &func, const double x1, const double x2, const double xacc) {
	const int maxit = 60;
	double fl = func(x1);
	double fh = func(x2);
	if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0)) {
		double xl = x1;
		double xh = x2;
		double ans = -9.99e99;
		for (int j = 0; j < maxit; j++) {
			double xm = 0.5 * (xl + xh);
			double fm = func(xm);
			double s = sqrt(fm * fm - fl * fh);
			if (s == 0.0) return ans;
			double xnew = xm + (xm - xl) * ((fl >= fh ? 1.0 : -1.0) * fm / s);
			if (fabs(xnew - ans) <= xacc) return ans;
			ans = xnew;
			double fnew = func(ans);
			if (fnew == 0.0) return ans;
			if (Sign(fm, fnew) != fm) {
				xl = xm;
				fl = fm;
				xh = ans;
				fh = fnew;
			}
			else if (Sign(fl, fnew) != fl) {
				xh = ans;
				fh = fnew;
			}
			else if (Sign(fh, fnew) != fh) {
				xl = ans;
				fl = fnew;
			}
			if (fabs(xh - xl) <= xacc) return ans;
		}
		throw("ridders method exceed maximum iterations");
	}
	else {
		if (fl == 0.0) return x1;
		if (fh == 0.0) return x2;
		throw ("root must be bracketed in riddlers algorithm.");
	}
}