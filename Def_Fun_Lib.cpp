#include "Fun_Lib.h"


int gfact(int num) {
	if (num == 0) return 1;
	else if (num == 1) return gfact(0);
	else return num * gfact(num - 1);
}


double trapz_e(double a, double b, pfn fg, int n) {
	// a-b --> interval limits
	// fg  --> given function
	// n   --> subintervals
	double sum_int = fg(a) * 0.5, h = (b - a) / n;
	for (int i = 1; i < n; i++) { sum_int += fg(a + i * h); }
	sum_int += fg(b) * 0.5;
	return sum_int * h;
}


double simps(double a, double b, pfn fg, int n) {
	// a-b --> interval limits
	// fg  --> given function
	// n   --> subintervals
	double sum_int = 0.0, inter_val = 0.0, h = (b - a) / n;
	for (int i = 0; i < n; i += 2) {
		inter_val = ((fg(a + h * i) + 4 * fg(a + h * (i + 1)) + fg(a + h * (i + 2))) / 3) * h;
		sum_int += inter_val;
	}
	return sum_int;
}


double pol_hor(Vec<double> cf, double x, int n) {
	double un = cf[n];
	for (int i = n - 1; i >= 0; i--) {
		un = un * x + cf[i];
	}
	return un;
}
