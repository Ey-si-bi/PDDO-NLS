#include "Peri1DNLS.h"


void Peri1DNLS::Peri1DMesh(double Length, int Matpt_x, int MaxFM,
	double HorizonRadCoeff) {
	system("mkdir Results");
	L = Length;
	Smax = MaxFM;
	DeltaC = HorizonRadCoeff;
	Mpx = Matpt_x;
	Delx = L / (Mpx - 1);
	Delv = Delx;
	Delta = DeltaC * Delx;
	SysSize = Mpx;			// Size of the Global System.
	NNZCoeff = 0;
	vector<Point<1>> Coords_L;
	vector<Point<1>> Coords_R;
	Coords.resize(Mpx);
	Coords_L.resize(Mpx);
	Coords_R.resize(Mpx);
	// Generating Material Point Coordinates.
	for (int i = 0; i < Mpx; i++) {
		Coords[i] = Point<1>(i * Delx - L / 2);
		Coords_L[i] = Point<1>(i * Delx - 3 * L / 2);
		Coords_R[i] = Point<1>(i * Delx + L / 2);
	}
	Coords_L[Mpx - 1] = Point<1>(1E+6);
	Coords_R[0] = Point<1>(1E+6);
	KDtree<1> PD_KdTree(Coords);
	KDtree<1> PD_KdTree_L(Coords_L);
	KDtree<1> PD_KdTree_R(Coords_R);
	NumFam.resize(Mpx);
	XiVecFam.resize(Mpx);
	Point<1> xi_vec;
	vector<int> Temp_Ptlst(Smax);				// Temporary Vector for Family Member Indices.
	vector<int> Temp_Ptlst_L(Smax);				// Temporary Vector for Family Member Indices.
	vector<int> Temp_Ptlst_R(Smax);				// Temporary Vector for Family Member Indices.
	int Pfam;									// Integer Indicating the Amount of Points inside a Family.
	int Pfam_L;									// Integer Indicating the Amount of Points inside a Family.
	int Pfam_R;									// Integer Indicating the Amount of Points inside a Family.
	// Generating Family Members Using KDTree Algorithm.
	for (int k = 0; k < Mpx; k++) {
		vector<int> Indx_Str(Smax);
		vector<Point<1>> Indx_Xi(Smax);
		Pfam_L = PD_KdTree_L.locatenear(Coords[k], Delta, Temp_Ptlst_L.data(), Smax);
		Pfam = PD_KdTree.locatenear(Coords[k], Delta, Temp_Ptlst.data(), Smax);
		Pfam_R = PD_KdTree_R.locatenear(Coords[k], Delta, Temp_Ptlst_R.data(), Smax);
		NNZCoeff += Pfam + Pfam_L + Pfam_R;
		NumFam[k].resize(Pfam + Pfam_L + Pfam_R);
		Indx_Str.resize(Pfam + Pfam_L + Pfam_R);
		Indx_Xi.resize(Pfam + Pfam_L + Pfam_R);
		std::iota(std::begin(Indx_Str), std::end(Indx_Str), 0);
		XiVecFam[k].resize(Pfam + Pfam_L + Pfam_R);
		int cnt = 0;
		for (int ii = 0; ii < Pfam_L; ii++) {
			NumFam[k][cnt] = Temp_Ptlst_L[ii];
			xi_vec = Vecdist(Coords_L[Temp_Ptlst_L[ii]], Coords[k]);
			XiVecFam[k][cnt] = xi_vec;
			cnt++;
		}
		for (int ii = 0; ii < Pfam; ii++) {
			NumFam[k][cnt] = Temp_Ptlst[ii];
			xi_vec = Vecdist(Coords[Temp_Ptlst[ii]], Coords[k]);
			XiVecFam[k][cnt] = xi_vec;
			cnt++;
		}
		for (int ii = 0; ii < Pfam_R; ii++) {
			NumFam[k][cnt] = Temp_Ptlst_R[ii];
			xi_vec = Vecdist(Coords_R[Temp_Ptlst_R[ii]], Coords[k]);
			XiVecFam[k][cnt] = xi_vec;
			cnt++;
		}
		// Sorting Family Member Indices.
		vector< pair <int, int> > vect;
		for (int i = 0; i < Pfam + Pfam_L + Pfam_R; i++)
			vect.push_back(make_pair(NumFam[k][i], Indx_Str[i]));
		sort(vect.begin(), vect.end());
		for (int i = 0; i < Pfam + Pfam_L + Pfam_R; i++) {
			NumFam[k][i] = vect[i].first;
			Indx_Xi[i] = XiVecFam[k][vect[i].second];
		}
		XiVecFam[k] = Indx_Xi;
	}
}


void Peri1DNLS::Peri1DSysInt() {
	NNZ = NNZCoeff;
	INCSOL.Resize(SysSize * 2);
	SOL.Resize(SysSize * 2);
	BMat.Resize(SysSize * 2, SysSize * 2, NNZ * 4);
}


void Peri1DNLS::Gensys(double DT, double TT,
	pfn InC) {
	Delt = DT;
	TTM = TT;
	TSN = TT / DT;
	IntCond = InC;
	for (int k = 0; k < Mpx; k++) {
		for (auto it = NumFam[k].begin(); it != NumFam[k].end(); ++it) {
			BMat.AddElemS(k, *it, 1.0);
		}
		for (auto it = NumFam[k].begin(); it != NumFam[k].end(); ++it) {
			BMat.AddElemS(k, *it + Mpx, 1.0);
		}
		SOL[k] = IntCond(Coords[k].x[0]);
	}
	for (int k = Mpx; k < 2 * Mpx; k++) {
		for (auto it = NumFam[k - Mpx].begin(); it != NumFam[k - Mpx].end(); ++it) {
			BMat.AddElemS(k, *it, 1.0);
		}
		for (auto it = NumFam[k - Mpx].begin(); it != NumFam[k - Mpx].end(); ++it) {
			BMat.AddElemS(k, *it + Mpx, 1.0);
		}
	}
}


double Wfunc(double ksi, double Delta) {
	return exp(-4.0 * (fabs(ksi) / Delta) * (fabs(ksi) / Delta));
}


Mat<double> GenBMat() {// Only For 1D NLS Equation.
	Mat<double> bmat(3, 3);
	bmat.SetNull();
	bmat(0, 0) = 1.0;
	bmat(1, 1) = 1.0;
	bmat(2, 2) = 2.0;
	return bmat;
}


Mat<double> GenAmat(Point<1> xi_vec, double Delta) {// Only for 1D Third Order Expansion. 
	double xi_mag = abs(xi_vec.x[0]);
	double xi_1 = xi_vec.x[0];
	Vec<double> Avec(3);
	Avec[0] = 1.0;
	Avec[1] = xi_1;
	Avec[2] = xi_1 * xi_1;
	double w = Wfunc(xi_mag, Delta);
	Mat<double> Amat = Outer(Avec, Avec);
	Amat.MultScal(w);
	return Amat;
}


Vec<double> InvoGFunc(Point<1> xi_vec, double Delta, Mat<double> amat) {// Only For Laplace Equation.
	Vec<double> GFuns(3);
	double xi_mag = abs(xi_vec.x[0]);
	double xi_1 = xi_vec.x[0];
	Vec<double> Avec(3);
	Avec[0] = 1.0;
	Avec[1] = xi_1;
	Avec[2] = xi_1 * xi_1;
	double w = Wfunc(xi_mag, Delta);
	for (int k = 0; k < 3; k++) {
		Vec<double> acol = amat[k];
		double C = Inner(acol, Avec);
		C *= w;
		GFuns[k] = C;
	}
	return GFuns;
}


Vec<double> FCalc(Vec<double> DelU, int SysSize, int Mpx,
	vector<vector<int>> NumFam,
	vector<Point<1>> Coords,
	vector<vector<Point<1>>> XiVecFam,
	double Delta, Vec<double> SOL,
	double Delt, double Delv) {
	Vec<double> F(SysSize * 2);
	F.SetValSca(0.0);
	Mat<double> Amat(3, 3);
	int cnt;
	for (int k = 0; k < Mpx; k++) {
		double temp_cf1 = 0.0, temp_cf2 = 0.0, temp_cf3 = 0.0, temp_cf4 = 0.0;
		Amat.SetNull();
		Mat<double> bmat = GenBMat();
		cnt = 0;
		for (auto it_1 = NumFam[k].begin(); it_1 != NumFam[k].end(); ++it_1) {
			Amat += GenAmat(XiVecFam[k][cnt], Delta);
			cnt++;
		}
		Amat.MultScal(Delv);
		Mat<double> amat = LUFactSolMult(Amat, bmat, 3);
		cnt = 0;
		for (auto it_2 = NumFam[k].begin(); it_2 != NumFam[k].end(); ++it_2) {
			Vec<double> GFunc = InvoGFunc(XiVecFam[k][cnt], Delta, amat);
			cnt++;
			temp_cf1 += GFunc[2] * (SOL[*it_2 + Mpx] + DelU[*it_2 + Mpx]) * Delv;
			temp_cf2 += GFunc[0] * (SOL[*it_2 + Mpx] + DelU[*it_2 + Mpx]) * Delv;
			temp_cf3 += GFunc[2] * (SOL[*it_2] + DelU[*it_2]) * Delv;
			temp_cf4 += GFunc[0] * (SOL[*it_2] + DelU[*it_2]) * Delv;
		}
		F[k] = DelU[k] + Delt * (0.5 * temp_cf1) + Delt * (temp_cf4 * temp_cf4 + temp_cf2 * temp_cf2) * temp_cf2;
		F[k + Mpx] = DelU[k + Mpx] - Delt * (0.5 * temp_cf3) - Delt * (temp_cf4 * temp_cf4 + temp_cf2 * temp_cf2) * temp_cf4;
	}
	return F;
}


void ForceCalcJacob1(int n, adept::adouble* x, int m, adept::adouble* f, void *GFDVDT) {
	adouble temp_cf1, temp_cf2, temp_cf3, temp_cf4;
	int st = n / 2;
	for (int i = 0; i < st; i++) {
		temp_cf1 += ((F_inf *)GFDVDT)->Gfuns(i, 2) * x[i + st] * ((F_inf *)GFDVDT)->Delv;
		temp_cf2 += ((F_inf *)GFDVDT)->Gfuns(i, 0) * x[i + st] * ((F_inf *)GFDVDT)->Delv;
		temp_cf3 += ((F_inf *)GFDVDT)->Gfuns(i, 2) * x[i] * ((F_inf *)GFDVDT)->Delv;
		temp_cf4 += ((F_inf *)GFDVDT)->Gfuns(i, 0) * x[i] * ((F_inf *)GFDVDT)->Delv;
	}
	for (int j = 0; j < m; j++) {
		f[j] = x[((F_inf *)GFDVDT)->Pint] + ((F_inf *)GFDVDT)->Delt * (0.5 * temp_cf1) + ((F_inf *)GFDVDT)->Delt * (temp_cf4 * temp_cf4 + temp_cf2 * temp_cf2) * temp_cf2;
	}
}


void ForceCalcJacob2(int n, adept::adouble* x, int m, adept::adouble* f, void *GFDVDT) {
	adouble temp_cf1, temp_cf2, temp_cf3, temp_cf4;
	int st = n / 2;
	for (int i = st; i < n; i++) {
		temp_cf1 += ((F_inf *)GFDVDT)->Gfuns(i - st, 2) * x[i] * ((F_inf *)GFDVDT)->Delv;
		temp_cf2 += ((F_inf *)GFDVDT)->Gfuns(i - st, 0) * x[i] * ((F_inf *)GFDVDT)->Delv;
		temp_cf3 += ((F_inf *)GFDVDT)->Gfuns(i - st, 2) * x[i - st] * ((F_inf *)GFDVDT)->Delv;
		temp_cf4 += ((F_inf *)GFDVDT)->Gfuns(i - st, 0) * x[i - st] * ((F_inf *)GFDVDT)->Delv;
	}
	for (int j = 0; j < m; j++) {
		f[j] = x[((F_inf *)GFDVDT)->Pint + st] - ((F_inf *)GFDVDT)->Delt * (0.5 * temp_cf3) - ((F_inf *)GFDVDT)->Delt * (temp_cf4 * temp_cf4 + temp_cf2 * temp_cf2) * temp_cf4;
	}
}


void JacobCalc(SMat& BMat, Vec<double> SOL,
	int SysSize, int Mpx,
	vector<vector<int>> NumFam,
	vector<Point<1>> Coords,
	vector<vector<Point<1>>> XiVecFam,
	double Delta, double Delt,
	double Delv, Vec<double> DelU) {
	F_inf Gfuns_Dt_Dv;
	Gfuns_Dt_Dv.Delt = Delt;
	Gfuns_Dt_Dv.Delv = Delv;
	Mat<double> Amat(3, 3);
	int cnt;
	for (int k = 0; k < Mpx; k++) {
		Mat<double> bmat = GenBMat();
		Gfuns_Dt_Dv.Pint = ValIndxFndrE(NumFam[k], k);
		cnt = 0;
		Amat.SetNull();
		for (auto it_1 = NumFam[k].begin(); it_1 != NumFam[k].end(); ++it_1) {
			Amat += GenAmat(XiVecFam[k][cnt], Delta);
			cnt++;
		}
		Mat<double> GFUNSMAT(cnt, 3);
		Amat.MultScal(Delv);
		Mat<double> amat = LUFactSolMult(Amat, bmat, 3);
		cnt = 0;
		for (auto it_2 = NumFam[k].begin(); it_2 != NumFam[k].end(); ++it_2) {
			Vec<double> GFunc = InvoGFunc(XiVecFam[k][cnt], Delta, amat);
			GFUNSMAT(cnt, 0) = GFunc[0];
			GFUNSMAT(cnt, 1) = GFunc[1];
			GFUNSMAT(cnt, 2) = GFunc[2];
			cnt++;
		}
		Vec<double> fjac(2 * cnt);
		Gfuns_Dt_Dv.Gfuns = GFUNSMAT;
		Vec<double> x(cnt * 2);
		Vec<double> x_u(cnt);
		Vec<double> x_v(cnt);
		Gfuns_Dt_Dv.SOL = ExtVec(SOL, *NumFam[k].begin(), *NumFam[k].begin() + cnt - 1);
		x_u = ExtVecF(DelU, NumFam[k], 0) + ExtVecF(SOL, NumFam[k], 0);
		x_v = ExtVecF(DelU, NumFam[k], Mpx) + ExtVecF(SOL, NumFam[k], Mpx);
		x = AppVec(x_u, x_v, cnt, cnt);
		JacobCalcVecFAD(cnt * 2, x, 1, &Gfuns_Dt_Dv, ForceCalcJacob1, fjac);
		AssgnValCsrV(BMat, fjac, k);
	}
	for (int k = Mpx; k < 2 * Mpx; k++) {
		Mat<double> bmat = GenBMat();
		Gfuns_Dt_Dv.Pint = ValIndxFndrE(NumFam[k - Mpx], k - Mpx);
		cnt = 0;
		Amat.SetNull();
		for (auto it_1 = NumFam[k - Mpx].begin(); it_1 != NumFam[k - Mpx].end(); ++it_1) {
			Amat += GenAmat(XiVecFam[k - Mpx][cnt], Delta);
			cnt++;
		}
		Mat<double> GFUNSMAT(cnt, 3);
		Amat.MultScal(Delv);
		Mat<double> amat = LUFactSolMult(Amat, bmat, 3);
		cnt = 0;
		for (auto it_2 = NumFam[k - Mpx].begin(); it_2 != NumFam[k - Mpx].end(); ++it_2) {
			Vec<double> GFunc = InvoGFunc(XiVecFam[k - Mpx][cnt], Delta, amat);
			GFUNSMAT(cnt, 0) = GFunc[0];
			GFUNSMAT(cnt, 1) = GFunc[1];
			GFUNSMAT(cnt, 2) = GFunc[2];
			cnt++;
		}
		Vec<double> fjac(2 * cnt);
		Gfuns_Dt_Dv.Gfuns = GFUNSMAT;
		Vec<double> x(cnt * 2);
		Vec<double> x_u(cnt);
		Vec<double> x_v(cnt);
		Gfuns_Dt_Dv.SOL = ExtVec(SOL, *NumFam[k - Mpx].begin(), *NumFam[k - Mpx].begin() + cnt - 1);
		x_u = ExtVecF(DelU, NumFam[k - Mpx], 0) + ExtVecF(SOL, NumFam[k - Mpx], 0);
		x_v = ExtVecF(DelU, NumFam[k - Mpx], Mpx) + ExtVecF(SOL, NumFam[k - Mpx], Mpx);
		x = AppVec(x_u, x_v, cnt, cnt);
		JacobCalcVecFAD(cnt * 2, x, 1, &Gfuns_Dt_Dv, ForceCalcJacob2, fjac);
		AssgnValCsrV(BMat, fjac, k);
	}
	BMat.UpdtHandle();
}


void Peri1DNLS::Peri1DSolve(double Tolerence, int MaximumIt) {
	Eps = Tolerence;
	Maxit = MaximumIt;
	BMat.CrtBLSmtrxCoo();
	BMat.CnvrtCoo2Csr();
	BMat.ChangeOneBase("CSR");
	INCSOL.SetValSca(0.0);
	JacobCalc(BMat, SOL, SysSize, Mpx,
		NumFam, Coords, XiVecFam, Delta, Delt,
		Delv, INCSOL);
	Vec<double> F1(SysSize * 2);
	Vec<double> F2(SysSize * 2);
	Vec<double> SysUpdt(SysSize * 2);
	Vec<double> RowVec;
	int cnt = 0;
	int tstpc = 0;
	double denom, num;
	double err = 1.0e99;
	F1 = FCalc(INCSOL, SysSize, Mpx, NumFam,
		Coords, XiVecFam, Delta, SOL, Delt, Delv);
	F1.MultScal(-1.0);
	err = F1.EucNorm();
	cout << err << endl;
	for (int k = 0; k < TSN; k++) {
	BEGINIT:
		for (int kk = 0; kk < Maxit; kk++) {
			INCSOL = DrcSol(BMat, F1);
			F2 = FCalc(INCSOL, SysSize, Mpx, NumFam,
				Coords, XiVecFam, Delta, SOL, Delt, Delv);
			err = F2.EucNorm();
			cout << err << endl;
			SysUpdt = F2 + F1 - SMatVecMultp(BMat, INCSOL);
			for (int ii = 0; ii < Mpx; ii++) {
				RowVec = ExtrVec4CsrRowCp1(BMat, INCSOL, ii, NumFam);
				denom = Inner(RowVec, RowVec);
				num = SysUpdt[ii];
				if (denom == 0.0) continue;
				RowVec.MultScal(num / denom);
				IncrValCsrV(BMat, RowVec, ii);
			}
			for (int ii = Mpx; ii < 2 * Mpx; ii++) {
				RowVec = ExtrVec4CsrRowCp2(BMat, INCSOL, ii, NumFam, Mpx);
				denom = Inner(RowVec, RowVec);
				num = SysUpdt[ii];
				if (denom == 0.0) continue;
				RowVec.MultScal(num / denom);
				IncrValCsrV(BMat, RowVec, ii);
			}
			BMat.UpdtHandle();
			cnt++;
			if (err < Eps) break;
		}
		if (cnt == Maxit) {
			INCSOL.SetValSca(0.0);
			JacobCalc(BMat, SOL,
				SysSize, Mpx, NumFam,
				Coords, XiVecFam, Delta, Delt,
				Delv, INCSOL);
			F1 = FCalc(INCSOL, SysSize, Mpx, NumFam,
				Coords, XiVecFam, Delta, SOL, Delt, Delv);
			F1.MultScal(-1.0);
			goto BEGINIT;
		}
		cnt = 0;
		tstpc++;
		SOL += INCSOL;
		PeriDerGen(SOL, SysSize, Mpx, NumFam, Coords, XiVecFam, Delta, Delv, tstpc);
		INCSOL.MultScal(1 / Delt);
		INCSOL.Write2File("Results\\NLS_Eqn T = " + to_string(tstpc) + "xDt(VeloxDt).dat");
		//SOL.Write2File("Results\\NLS_Eqn T = " + to_string(tstpc) + "xDt(UDt).dat");
		INCSOL.SetValSca(0.0);
		F1 = FCalc(INCSOL, SysSize, Mpx, NumFam,
			Coords, XiVecFam, Delta, SOL, Delt, Delv);
		F1.MultScal(-1.0);
	}
	SOL.Resize(Mpx);
	
}


void PeriDerGen(Vec<double> SOL,
	int SysSize, int Mpx,
	vector<vector<int>> NumFam,
	vector<Point<1>> Coords,
	vector<vector<Point<1>>> XiVecFam,
	double Delta, double Delv, int tstpc){
	Vec<double> U(SysSize);
	Vec<double> V(SysSize);
	Vec<double> U_x(SysSize);
	Vec<double> V_x(SysSize);
	Vec<double> U_xx(SysSize);
	Vec<double> V_xx(SysSize);
	Vec<double> abs_h(SysSize);
	Mat<double> Amat(3, 3);
	int cnt;
	for (int k = 0; k < Mpx; k++) {
		double temp_cf1 = 0.0, temp_cf2 = 0.0, temp_cf3 = 0.0, temp_cf4 = 0.0, temp_cf5 = 0.0, temp_cf6 = 0.0;
		Amat.SetNull();
		Mat<double> bmat = GenBMat();
		cnt = 0;
		for (auto it_1 = NumFam[k].begin(); it_1 != NumFam[k].end(); ++it_1) {
			Amat += GenAmat(XiVecFam[k][cnt], Delta);
			cnt++;
		}
		Amat.MultScal(Delv);
		Mat<double> amat = LUFactSolMult(Amat, bmat, 3);
		cnt = 0;
		for (auto it_2 = NumFam[k].begin(); it_2 != NumFam[k].end(); ++it_2) {
			Vec<double> GFunc = InvoGFunc(XiVecFam[k][cnt], Delta, amat);
			cnt++;
			temp_cf1 += GFunc[0] * (SOL[*it_2]) * Delv;
			temp_cf2 += GFunc[1] * (SOL[*it_2]) * Delv;
			temp_cf3 += GFunc[2] * (SOL[*it_2]) * Delv;
			temp_cf4 += GFunc[0] * (SOL[*it_2 + Mpx]) * Delv;
			temp_cf5 += GFunc[1] * (SOL[*it_2 + Mpx]) * Delv;
			temp_cf6 += GFunc[2] * (SOL[*it_2 + Mpx]) * Delv;
		}
		U[k] = temp_cf1;
		U_x[k] = temp_cf2;
		U_xx[k] = temp_cf3;
		V[k] = temp_cf4;
		V_x[k] = temp_cf5;
		V_xx[k] = temp_cf6;
		abs_h[k] = pow(temp_cf1 * temp_cf1 + temp_cf4 * temp_cf4, 0.5);
	}
	U.Write2File("Results\\U T = " + to_string(tstpc) + "xDt.dat");
	U_x.Write2File("Results\\Ux T = " + to_string(tstpc) + "xDt.dat");
	U_xx.Write2File("Results\\Uxx T = " + to_string(tstpc) + "xDt.dat");
	V.Write2File("Results\\V T = " + to_string(tstpc) + "xDt.dat");
	V_x.Write2File("Results\\Vx T = " + to_string(tstpc) + "xDt.dat");
	V_xx.Write2File("Results\\Vxx T = " + to_string(tstpc) + "xDt.dat");
	abs_h.Write2File("Results\\H = " + to_string(tstpc) + "xDt.dat");
}


int ValIndxFndrE(vector<int> vec, int val) {
	for (int i = 0; i < vec.size(); i++) {
		if (vec.at(i) == val) {
			return i;
		}
	}
}