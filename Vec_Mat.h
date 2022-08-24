/*******************************************************************************
* Ali Can Bekar 2019.
*
* This software is a sparse-dense matrix and vector library. It is a wrapper
* for mkl's lapack, blas and sblas libraries. Row-major order is followed for
* matrices and standart vector library from c++ is used.
*******************************************************************************/
#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <mkl.h>
#include "adept.h"
#include "Comp_Geo.h"
using namespace std;
using adept::adouble;

// FORWARD DECLERATIONS
/*******************************************************************************/
template<typename T> class Mat;
template<typename T> class Vec;
class SMat;
template<typename T> using pfnT = T(*)(int);
template<typename T> using pfn2T = T(*)(int, int);
template<typename T> Mat<T> Transpose(Mat<T>& M);
template<typename T> Mat<T> ExtMat(Mat<T>& M, int rows, int rowe, int cols, int cole);
template<typename T> Mat<T> MatMult(Mat<T>& M1, Mat<T>& M2);
template<typename T> Vec<T> RVecM(Mat<T>& M, int i);
template<typename T> Vec<T> ExtVec(Vec<T>& V, int el_s, int el_e);
template<typename T> Vec<T> operator+(const Vec<T>& V1, const Vec<T>& V2);
template<typename T> Vec<T> operator-(const Vec<T>& V1, const Vec<T>& V2);
template<typename T> Vec<T> operator*(const Vec<T>& V1, const Vec<T>& V2);
template<typename T> Mat<T> operator+(const Mat<T>& M1, const Mat<T>& M2);
template<typename T> Mat<T> operator-(const Mat<T>& M1, const Mat<T>& M2);
template<typename T> Mat<T> Outer(Vec<T>& V1, Vec<T>& V2);
static double Inner(Vec<double>& V1, Vec<double>& V2);
static Vec<double> MatVecMultp(Mat<double>& M, Vec<double>& V);
static Vec<double> LUFactSol(Mat<double>& M, Vec<double>& R);
static void IncrValCsrV(SMat& SCM, Vec<double>& V, int r_ind);
static void AssgnValCsrV(SMat& SCM, Vec<double> V, int r_ind);
static Vec<double> SMatVecMultp(SMat& SCM, Vec<double>& V);
static Vec<double> ExtrVec4CsrRow(SMat& SCM, Vec<double>& V, int Rownum, vector<vector<int>> NumFam);
static void JacobCalcVecF(void* Usdfc, int n, int m, Vec<double>& fjac, Vec<double> x, double eps, void* Usrdt);
static Vec<double> DrcSol(SMat& SCM, Vec<double>& RHS);
static Vec<double> GMRES_Pc_ILU0(SMat& SCM, Vec<double>& RHS, Vec<double>& Init);
static Vec<double> GMRES_NPc(SMat& SCM, Vec<double>& RHS, Vec<double>& Init);
typedef void(*JC)(int n, adept::adouble* x, int m, adept::adouble* f, void *GFDVDT);
static void JacobCalcVecFAD(int n, Vec<double> x, int m, void* Usrdt, JC Usdfc, Vec<double>& Jac);
static Mat<double> LUFactSolMult(Mat<double>& M, Mat<double>& R, int SizeR);
static Mat<double> Inv(Mat<double>& M);
static void IncrValCsrV(SMat& SCM, Vec<double>& V, int r_ind);
static void AssgnValCsrV(SMat& SCM, Vec<double> V, int r_ind);
/*******************************************************************************/



// VECTOR CLASS DECLERATIONS
/*******************************************************************************/
template<typename T>
class Vec {
private:
	vector<T> vec;
	int size;
public:
	Vec<T>() : size(NULL), vec(NULL, 0) {};
	Vec<T>(int len) : size(len), vec(len, 0) {};
	Vec<T>(int len, vector<T> predfvec);
	Vec<T>(const Vec<T>& VC);
	Vec<T>& operator=(const Vec<T>& VC);
	Vec<T>& operator+=(const Vec<T>& VC);
	Vec<T>& operator-=(const Vec<T>& VC);
	T& operator[](int i);
	void Resize(int dim);
	int ElCnt();
	int ValIndxFndr(T val);
	void SetValFunc(pfnT<T> func);
	void ElSwap(int i, int j);
	void SetValSca(T val);
	void MultScal(T d);
	void ReadFile(string Fname);
	void Write2File(string Fname);
	T SumEl();
	T OneNorm();
	double EucNorm();
	T MaxNorm();
	int NonZElCnt();
	void PrintEl(int elnum);
	void PrintAll();
	friend Vec<double> MatVecMultp(Mat<double>& M, Vec<double>& V);
	friend Vec ExtVec<T>(Vec& V, int el_s, int el_e);
	friend Vec operator+<T>(const Vec& V1, const Vec& V2);
	friend Vec operator-<T>(const Vec& V1, const Vec& V2);
	friend Vec operator*<T>(const Vec& V1, const Vec& V2);
	friend Mat<T> Outer<T>(Vec& V1, Vec& V2);
	friend double Inner(Vec<double>& V1, Vec<double>& V2);
	friend Vec<double> LUFactSol(Mat<double>& M, Vec<double>& R);
	friend void IncrValCsrV(SMat& SCM, Vec<double>& V, int r_ind);
	friend void AssgnValCsrV(SMat& SCM, Vec<double> V, int r_ind);
	friend Vec<double> SMatVecMultp(SMat& SCM, Vec<double>& V);
	friend Vec<double> ExtrVec4CsrRow(SMat& SCM, Vec<double>& V, int Rownum, vector<vector<int>> NumFam);
	friend void JacobCalcVecF(void* Usdfc, int n, int m, Vec<double>& fjac, Vec<double> x, double eps, void* Usrdt);
	friend void JacobCalcVecFAD(int n, Vec<double> x, int m, void* Usrdt, JC Usdfc, Vec<double>& Jac);
	friend Vec<double> DrcSol(SMat& SCM, Vec<double>& RHS);
	friend Vec<double> GMRES_NPc(SMat& SCM, Vec<double>& RHS, Vec<double>& Init);
	friend Vec<double> GMRES_Pc_ILU0(SMat& SCM, Vec<double>& RHS, Vec<double>& Init);
};
/*******************************************************************************/



// MATRIX CLASS DECLERATIONS
/*******************************************************************************/
template<typename T>
class Mat {
private:
	vector<T> mat;
	int rownum;
	int colnum;
public:
	Mat<T>() : colnum(NULL), rownum(NULL), mat(NULL, 0) {};
	Mat<T>(int rn, int cn) : rownum(rn), colnum(cn), mat(rn * cn, 0) {};
	Mat<T>(int rn, int cn, vector<T> predfmat);
	Mat<T>(const Mat<T>& MC);
	Mat<T>& operator=(const Mat<T>& MC);
	Mat<T>& operator+=(const Mat<T>& MC);
	Mat<T>& operator-=(const Mat<T>& MC);
	Vec<T> operator[](int i);
	T& operator()(int i, int j);
	void Resize(int dimr, int dimc);
	void SetValFunc(pfn2T<T> func);
	void RowSwap(int i, int j);
	void MultScal(T d);
	void MultScalRow(int i, T d);
	void MultScalCol(int i, T d);
	void ReadFile(string Fname);
	void SetNull();
	void SetEye();
	double OneNorm();
	double MaxNorm();
	double FrobNorm();
	double InfNorm();
	double Det();
	void PrintEl(int rn, int cn);
	void PrintAll();
	friend Mat Transpose<T>(Mat& M);
	friend Mat ExtMat<T>(Mat& M, int rows, int rowe, int cols, int cole);
	friend Mat MatMult<T>(Mat& M1, Mat& M2);
	friend Vec<T> RVecM<T>(Mat& M, int i);
	friend Vec<double> MatVecMultp(Mat<double>& M, Vec<double>& V);
	friend Mat operator+<T>(const Mat& M1, const Mat& M2);
	friend Mat operator-<T>(const Mat& M1, const Mat& M2);
	friend Mat Outer<T>(Vec<T>& V1, Vec<T>& V2);
	friend Vec<double> LUFactSol(Mat<double>& M, Vec<double>& R);
	friend Mat<double> LUFactSolMult(Mat<double>& M, Mat<double>& R, int SizeR);
	friend Mat<double> Inv(Mat<double>& M);
};
/*******************************************************************************/



// SPARSE MATRIX CLASS DECLERATIONS
/*******************************************************************************/
class SMat {
private:
	vector<double> data;
	vector<int> rowind;
	vector<int> colind;
	sparse_matrix_t BLSmtrx;
	struct matrix_descr descrBLSmtrx;
	sparse_index_base_t IndxBs;
	int rownum;
	int colnum;
	int elnum;
public:
	SMat() : colnum(NULL), rownum(NULL), elnum(0), data(NULL, 0), rowind(NULL, 0), colind(NULL, 0), IndxBs(SPARSE_INDEX_BASE_ZERO) {};
	SMat(int rn, int cn, int nonz) : rownum(rn), colnum(cn), data(nonz, 0), rowind(nonz, 0), colind(nonz, 0), IndxBs(SPARSE_INDEX_BASE_ZERO) {};
	void Resize(int rn, int cn, int nonz) {
		rownum = rn;
		colnum = cn;
		elnum = 0;
		rowind.resize(nonz);
		colind.resize(nonz);
		data.resize(nonz);
	};
	void AddElemS(int r_ind, int c_ind, double val) {
		rowind[elnum] = r_ind;
		colind[elnum] = c_ind;
		data[elnum] = val;
		elnum++;
	};
	void CrtBLSmtrxCoo() {
		mkl_sparse_d_create_coo(&BLSmtrx, SPARSE_INDEX_BASE_ZERO, rownum, colnum, elnum, rowind.data(), colind.data(), data.data());
		descrBLSmtrx.type = SPARSE_MATRIX_TYPE_GENERAL;
	};
	void CnvrtCoo2Csr() {
		mkl_sparse_convert_csr(BLSmtrx, SPARSE_OPERATION_NON_TRANSPOSE, &BLSmtrx);
		vector<pair<int, pair<int, double>>> zipped;
		for (int i = 0; i < elnum; i++) {
			zipped.push_back(make_pair(rowind[i], make_pair(colind[i], data[i])));
		}
		std::sort(zipped.begin(), zipped.end());
		rowind.resize(rownum + 1);
		std::fill(rowind.begin(), rowind.end(), 0);
		for (int i = 0; i < elnum; i++) {
			rowind[zipped[i].first + 1] += 1;
			colind[i] = zipped[i].second.first;
			data[i] = zipped[i].second.second;
		}
		for (int i = 1; i < rownum + 1; i++) {
			rowind[i] += rowind[i - 1];
		}
	};
	void ChangeOneBase(string Mattp) {
		IndxBs = SPARSE_INDEX_BASE_ONE;
		if (Mattp.compare("COO") == 0) {
			for (int i = 0; i < elnum; i++) {
				colind[i] += 1;
				rowind[i] += 1;
			}
		}
		else if (Mattp.compare("CSR") == 0) {
			for (int i = 0; i < elnum; i++) {
				colind[i] += 1;
			}
			for (int i = 0; i < rownum + 1; i++) {
				rowind[i] += 1;
			}
		}
	};
	void UpdtHandle() {
		mkl_sparse_destroy(BLSmtrx);
		mkl_sparse_d_create_csr(&BLSmtrx, SPARSE_INDEX_BASE_ONE, rownum, colnum, rowind.data(), rowind.data() + 1, colind.data(), data.data());
		descrBLSmtrx.type = SPARSE_MATRIX_TYPE_GENERAL;
	};
	void FillOnesCsr() {
		std::fill(data.begin(), data.end(), 1.0);
	};
	void FillValCsrPeriSys(vector<vector<int>> NumFam, int SysSize) {
		for (int i = 0; i < SysSize; i++) {
			for (auto it = NumFam[i].begin(); it != NumFam[i].end(); ++it) {
				data[rowind[i] - 1 + *it] = 1.0;
			}
		}
	};
	friend void IncrValCsrV(SMat& SCM, Vec<double>& V, int r_ind);
	friend void AssgnValCsrV(SMat& SCM, Vec<double> V, int r_ind);
	friend Vec<double> SMatVecMultp(SMat& SCM, Vec<double>& V);
	friend Vec<double> ExtrVec4CsrRow(SMat& SCM, Vec<double>& V, int Rownum, vector<vector<int>> NumFam);
	friend Vec<double> DrcSol(SMat& SCM, Vec<double>& RHS);
	friend Vec<double> GMRES_NPc(SMat& SCM, Vec<double>& RHS, Vec<double>& Init);
	friend Vec<double> GMRES_Pc_ILU0(SMat& SCM, Vec<double>& RHS, Vec<double>& Init);
};
/*******************************************************************************/



// REGULAR FUNCTIONS
/*******************************************************************************/
inline void error(const string& s)
{
	throw runtime_error(s);
}
/*******************************************************************************/



// VECTOR CLASS FUNCTIONS
/*******************************************************************************/
template <typename T>
Vec<T>::Vec(int len, vector<T> predfvec) {
	size = len;
	vec = predfvec;
}


template <typename T>
Vec<T>::Vec(const Vec<T>& VC) {
	size = VC.size;
	vec = VC.vec;
}


template <typename T>
Vec<T>& Vec<T>::operator=(const Vec<T>& VC) {
	size = VC.size;
	if (this != &VC)
		vec = VC.vec;
	return *this;
}


template <typename T>
Vec<T>& Vec<T>::operator+=(const Vec<T>& VC) {
	if (size != VC.size) error("vector sizes don't match!");
	for (int i = 0; i < size; i++) {
		vec[i] += VC.vec.at(i);
	}
	return *this;
}


template <typename T>
Vec<T>& Vec<T>::operator-=(const Vec<T>& VC) {
	if (size != VC.size) error("vector sizes don't match!");
	for (int i = 0; i < size; i++) {
		vec[i] -= VC.vec.at(i);
	}
	return *this;
}


template <typename T>
T& Vec<T>::operator[](int i) {
	return vec[i];
}


template <typename T>
void Vec<T>::Resize(int dim) {
	vec.resize(dim);
	size = dim;
}


template <typename T>
int Vec<T>::ElCnt() {
	return size;
}


template <typename T>
int Vec<T>::ValIndxFndr(T val) {
	for (int i = 0; i < size; i++) {
		if (vec.at(i) == val) {
			return i;
		}
	}
	return -1;
}


template <typename T>
void Vec<T>::SetValFunc(pfnT<T> func) {
	for (int i = 0; i < size; i++) {
		vec.at(i) = func(i);
	}
}


template <typename T>
void Vec<T>::ElSwap(int i, int j) {
	double temp = vec.at(i);
	vec.at(i) = vec.at(j);
	vec.at(j) = temp;
}


template <typename T>
void Vec<T>::SetValSca(T val) {
	vec.assign(size, val);
}


template <typename T>
void Vec<T>::MultScal(T d) {
	for (int k = 0; k < size; k++) {
		vec.at(k) = d * vec.at(k);
	}
}


template <typename T>
void Vec<T>::ReadFile(string Fname) {
	double elm;
	ifstream myfile(Fname);

	if (myfile.is_open())
	{
		int index = 0;
		while (myfile >> elm)
		{
			vec.at(index++) = elm;
		}
		myfile.close();
	}

}


template <typename T>
void Vec<T>::Write2File(string Fname) {
	ofstream outfile;
	outfile.open(Fname);
	if (outfile.is_open())
	{
		int index = 0;
		while (size > index)
		{
			outfile.precision(12);
			outfile << vec.at(index++) << endl;
		}
		outfile.close();
	}
}


template <typename T>
T Vec<T>::SumEl() {
	T Sum = NULL;
	for (int i = 0; i < size; i++) {
		Sum += vec.at(i);
	}
	return Sum;
}


template <typename T>
T Vec<T>::OneNorm() {
	T norm = NULL;
	for (int i = 0; i < size; i++) {
		norm += fabs(vec.at(i));
	}
	return norm;
}


template <typename T>
double Vec<T>::EucNorm() {
	// Only for Vectors of Double Precision Members.
	int incx = 1;
	double norm2 = dnrm2(&size, vec.data(), &incx);
	return norm2;
}


template <typename T>
T Vec<T>::MaxNorm() {
	T max = fabs(vec.at(0));
	for (int i = 1; i < size; i++) {
		if (fabs(vec.at(i)) > max) {
			max = fabs(vec.at(i));
		}
	}
	return max;
}


template <typename T>
int Vec<T>::NonZElCnt() {
	int count = 0;
	for (int i = 0; i < size; i++) {
		if (vec.at(i) != 0) count += 1;
	}
	return count;
}


template <typename T>
void Vec<T>::PrintEl(int elnum) {
	cout << vec.at(elnum) << endl;
}


template <typename T>
void Vec<T>::PrintAll() {
	for (int i = 0; i < size; i++) {
		cout << vec.at(i) << endl;
	}
}
/*******************************************************************************/



// MATRIX CLASS FUNCTIONS
/*******************************************************************************/
template <typename T>
Mat<T>::Mat(int rn, int cn, vector<T> predfmat) {
	rownum = rn; colnum = cn;
	vector<T> mat(rn * cn);
	for (int i = 0; i < cn * rn; i++) {
		mat.at(i) = predfmat.at(i);
	}
}


template <typename T>
Mat<T>::Mat(const Mat<T>& MC) {
	rownum = MC.rownum;
	colnum = MC.colnum;
	mat = MC.mat;
}


template <typename T>
Mat<T>& Mat<T>::operator=(const Mat<T>& MC) {
	rownum = MC.rownum;
	colnum = MC.colnum;
	if (this != &MC)
		mat = MC.mat;
	return *this;
}


template <typename T>
Mat<T>& Mat<T>::operator+=(const Mat<T>& MC) {
	for (int i = 0; i < MC.colnum * MC.rownum; i++) {
		mat.at(i) += MC.mat.at(i);
	}
	return *this;
}


template <typename T>
Mat<T>& Mat<T>::operator-=(const Mat<T>& MC) {
	for (int i = 0; i < MC.colnum * MC.rownum; i++) {
		mat.at(i) -= MC.mat.at(i);
	}
	return *this;
}


template <typename T>
Vec<T> Mat<T>::operator[](int i) {
	Vec<T> VCol(rownum);
	for (int k = 0; k < rownum; k++) {
		VCol[k] = mat.at(k * colnum + i);
	}
	return VCol;
}


template <typename T>
T& Mat<T>::operator()(int i, int j) {
	return mat[i * colnum + j];
}


template <typename T>
void Mat<T>::Resize(int dimr, int dimc) {
	mat.resize(dimr * dimc);
	rownum = dimr;
	colnum = dimc;
}


template <typename T>
void Mat<T>::SetValFunc(pfn2T<T> func) {
	for (int i = 0; i < rownum; i++) {
		for (int k = 0; k < colnum; k++) {
			mat.at(i * colnum + k) = func(i, k);
		}
	}
}


template <typename T>
void Mat<T>::RowSwap(int i, int j) {
	double tempnum;
	for (int k = 0; k < colnum; k++) {
		tempnum = mat.at(i * colnum + k);
		mat.at(i * colnum + k) = mat.at(j * colnum + k);
		mat.at(j * colnum + k) = tempnum;
	}
}


template <typename T>
void Mat<T>::MultScal(T d) {
	for (int k = 0; k < colnum * rownum; k++) {
		mat.at(k) = d * mat.at(k);
	}
}


template <typename T>
void Mat<T>::MultScalRow(int i, T d) {
	for (int k = 0; k < colnum; k++) {
		mat.at(i * colnum + k) = d * mat.at(i * colnum + k);
	}
}


template <typename T>
void Mat<T>::MultScalCol(int i, T d) {
	for (int k = 0; k < rownum; k++) {
		mat.at(k * colnum + i) = d * mat.at(k * colnum + i);
	}
}


template <typename T>
void Mat<T>::ReadFile(string Fname) {
	double elm;
	ifstream myfile(Fname);

	if (myfile.is_open())
	{
		int index = 0;
		while (myfile >> elm)
		{
			mat.at(index++) = elm;
		}
		myfile.close();
	}

}


template <typename T>
void Mat<T>::SetNull() {
	mat.assign(colnum * rownum, 0.0);
}


template <typename T>
void Mat<T>::SetEye() {
	for (int i = 0; i < rownum; i++) {
		for (int k = 0; k < colnum; k++) {
			if (i != k) mat.at(i * colnum + k) = 0;
			else mat.at(i * colnum + k) = 1.0;
		}
	}
}


template <typename T>
double Mat<T>::OneNorm() {
	char normtype = '1';
	double onen = LAPACKE_dlange(LAPACK_ROW_MAJOR, normtype, rownum, colnum, mat.data(), rownum);
	return onen;
}


template <typename T>
double Mat<T>::MaxNorm() {
	char normtype = 'M';
	double max = LAPACKE_dlange(LAPACK_ROW_MAJOR, normtype, rownum, colnum, mat.data(), rownum);
	return max;
}


template <typename T>
double Mat<T>::FrobNorm() {
	char normtype = 'F';
	double frob = LAPACKE_dlange(LAPACK_ROW_MAJOR, normtype, rownum, colnum, mat.data(), rownum);
	return frob;
}


template <typename T>
double Mat<T>::InfNorm() {
	char normtype = 'I';
	double InfN = LAPACKE_dlange(LAPACK_ROW_MAJOR, normtype, rownum, colnum, mat.data(), rownum);
	return InfN;
}


template <typename T>
void Mat<T>::PrintEl(int rn, int cn) {
	cout << mat.at(rn * colnum + cn) << endl;
}


template <typename T>
void Mat<T>::PrintAll() {
	for (int i = 0; i < rownum; i++) {
		for (int k = 0; k < colnum; k++) {
			cout << mat.at(i * colnum + k);
			cout.width(6);
		}
		cout << endl;
		cout.width(0);
	}
}


template <typename T>
double Mat<T>::Det() {
	double det = 1.0;
	int info;
	int* ipv = new int[colnum];
	info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, rownum, colnum, mat.data(), rownum, ipv);
	for (int i = 0; i < colnum; i++) {
		det *= mat.at(i + i * colnum);
		if (ipv[i] = i + 1) det = -det;
	}
	return det;
}
/*******************************************************************************/


// FRIEND FUNCTIONS
/*******************************************************************************/
template <typename T>
Mat<T> Transpose(Mat<T>& M) {
	Mat<T> MT(M.colnum, M.rownum);
	for (int i = 0; i < M.rownum; i++) {
		for (int k = 0; k < M.colnum; k++) {
			MT.mat.at(k * M.rownum + i) = M.mat.at(i * M.colnum + k);
		}
	}
	return MT;
}


template <typename T>
Mat<T> ExtMat(Mat<T>& M, int rows, int rowe, int cols, int cole) {
	int row_num = rowe - rows + 1;
	int col_num = cole - cols + 1;
	Mat<T> SubM(row_num, col_num);
	for (int i = 0; i < row_num; i++) {
		for (int k = 0; k < col_num; k++) {
			SubM(i, k) = M(rows + i, cols + k);
		}
	}
	return SubM;
}


template <typename T>
Mat<T> MatMult(Mat<T>& M1, Mat<T>& M2) {
	Mat<T> MR(M1.rownum, M2.colnum);
	CBLAS_TRANSPOSE  transM1 = CblasNoTrans;
	CBLAS_TRANSPOSE  transM2 = CblasNoTrans;
	double alpha = 1.0, beta = 0.0;
	cblas_dgemm(CblasRowMajor, transM1, transM2, M1.rownum, M2.colnum, M1.colnum, alpha, M1.mat.data(), M1.rownum, M2.mat.data(), M2.rownum, beta, MR.mat.data(), MR.rownum);
	return MR;
}


template <typename T>
Vec<T> RVecM(Mat<T>& M, int i) {
	Vec<T> VRow(M.colnum);
	for (int k = 0; k < M.colnum; k++) {
		VRow[k] = M.mat.at(k + i * M.colnum);
	}
	return VRow;
}


template <typename T>
Vec<double> MatVecMultp(Mat<double>& M, Vec<double>& V) {
	Vec<double> Res(M.rownum);
	CBLAS_TRANSPOSE trans = CblasNoTrans;
	double alpha = 1.0, beta = 0.0;
	int incx = 1, incy = 1;
	cblas_dgemv(CblasRowMajor, trans, M.rownum, M.colnum, alpha, M.mat.data(), M.rownum, V.vec.data(), incx, beta, Res.vec.data(), incy);
	return Res;
}


template <typename T>
Vec<T> ExtVec(Vec<T>& V, int el_s, int el_e) {
	int sz = el_e - el_s + 1;
	Vec<T> SubV(sz);
	for (int i = 0; i < sz; i++) {
		SubV[i] = V[el_s + i];
	}
	return SubV;
}


template <typename T>
Vec<T> operator+(const Vec<T>& V1, const Vec<T>& V2) {
	if (V1.size != V2.size) error("vector sizes don't match!");
	Vec<T> V3(V1.size);
	for (int i = 0; i < V1.size; i++) {
		V3.vec[i] = V1.vec[i] + V2.vec[i];
	}
	return V3;
}


template <typename T>
Vec<T> operator-(const Vec<T>& V1, const Vec<T>& V2) {
	if (V1.size != V2.size) error("vector sizes don't match!");
	Vec<T> V3(V1.size);
	for (int i = 0; i < V1.size; i++) {
		V3.vec[i] = V1.vec[i] - V2.vec[i];
	}
	return V3;
}


template <typename T>
Vec<T> operator*(const Vec<T>& V1, const Vec<T>& V2) {
	if (V1.size != V2.size) error("vector sizes don't match!");
	Vec<T> V3(V1.size);
	for (int i = 0; i < V1.size; i++) {
		V3.vec[i] = V1.vec[i] * V2.vec[i];
	}
	return V3;
}


template <typename T>
Mat<T> operator+(const Mat<T>& M1, const Mat<T>& M2) {
	Mat<T> M3(M1.rownum, M1.colnum);
	for (int i = 0; i < M1.colnum * M1.rownum; i++) {
		M3.mat[i] = M1.mat[i] + M2.mat[i];
	}
	return M3;
}


template <typename T>
Mat<T> operator-(const Mat<T>& M1, const Mat<T>& M2) {
	Mat<T> M3(M1.rownum, M1.colnum);
	for (int i = 0; i < M1.colnum * M1.rownum; i++) {
		M3.mat[i] = M1.mat[i] - M2.mat[i];
	}
	return M3;
}


template <typename T>
Mat<T> Outer(Vec<T>& V1, Vec<T>& V2) {
	Mat<T> M(V1.size, V1.size);
	for (int i = 0; i < M.colnum; i++) {
		for (int k = 0; k < M.rownum; k++) {
			M.mat[i * M.rownum + k] = V1.vec[i] * V2.vec[k];
		}
	}
	return M;
}


static double Inner(Vec<double>& V1, Vec<double>& V2) {
	double inprd;
	int inc = 1;
	inprd = ddot(&V1.size, V1.vec.data(), &inc, V2.vec.data(), &inc);
	return inprd;
}


static Vec<double> LUFactSol(Mat<double>& M, Vec<double>& R) {
	int info;
	int numrs = 1;
	int* ipv = new int[M.colnum];
	info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, M.rownum, numrs, M.mat.data(), M.rownum, ipv, R.vec.data(), 1);
	Vec<double> SOL = R;
	return SOL;
}


static Mat<double> LUFactSolMult(Mat<double>& M, Mat<double>& R, int SizeR) {
	int info;
	int numrs = SizeR;
	int* ipv = new int[M.colnum];
	info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, M.rownum, numrs, M.mat.data(), M.rownum, ipv, R.mat.data(), R.colnum);
	Mat<double> SOL = R;
	return SOL;
}


static Mat<double> Inv(Mat<double>& M) {
	Mat<double> MInv(M.rownum, M.colnum);
	int info = M.colnum;
	char uplo = 'U';
	int* ipv = new int[M.colnum];
	info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, M.rownum, M.colnum, M.mat.data(), M.rownum, ipv);
	info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, M.rownum, M.mat.data(), M.rownum, ipv);
	delete ipv;
	MInv = M;
	return MInv;
}


static void IncrValCsrV(SMat& SCM, Vec<double>& V, int r_ind) {
	int cnt = 0;
	int eln = V.ElCnt();
	for (int i = SCM.rowind[r_ind] - 1; i < SCM.rowind[r_ind] - 1 + eln; i++) {
		SCM.data.at(i) += V[cnt];
		cnt++;
	}
}


static void AssgnValCsrV(SMat& SCM, Vec<double> V, int r_ind) {
	int cnt = 0;
	int eln = V.ElCnt();
	for (int i = SCM.rowind[r_ind] - 1; i < SCM.rowind[r_ind] - 1 + eln; i++) {
		SCM.data.at(i) = V[cnt];
		cnt++;
	}
}


static Vec<double> SMatVecMultp(SMat& SCM, Vec<double>& V) {
	MKL_INT NROWS = SCM.rownum;
	MKL_INT NCOLS = SCM.colnum;
	MKL_INT NNONZEROS = SCM.elnum;
	Vec<double> MultVec(SCM.rownum);
	double alpha = 1.0, beta = 0.0;
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		SCM.BLSmtrx,
		SCM.descrBLSmtrx,
		V.vec.data(),
		beta,
		MultVec.vec.data());
	return MultVec;
}


static Vec<double> ExtrVec4CsrRow(SMat& SCM, Vec<double>& V, int Rownum, vector<vector<int>> NumFam) {
	int eln = NumFam[Rownum].size();
	Vec<double> RowVec(eln);
	int cnt = 0;
	for (int i = SCM.rowind[Rownum] - 1; i < SCM.rowind[Rownum] - 1 + eln; i++) {
		RowVec[cnt] = V[SCM.colind[i] - 1];
		cnt++;
	}
	return RowVec;
}


static void JacobCalcVecF(void* Usdfc, int n, int m, Vec<double>& fjac, Vec<double> x, double eps, void* Usrdt) {
	djacobix((USRFCNXD)Usdfc, &n, &m, fjac.vec.data(), x.vec.data(), &eps, Usrdt);
}


static void JacobCalcVecFAD(int n, Vec<double> x, int m, void* Usrdt, JC Usdfc, Vec<double>& Jac) {
	using adept::adouble;
	adept::Stack stack;
	std::vector<adouble> X_a(n);
	set_values(&X_a[0], n, x.vec.data());
	stack.new_recording(); // Start recording
	std::vector<adouble> Y(m); // Create vector of active output variables
	Usdfc(n, &X_a[0], m, &Y[0], Usrdt); // Run algorithm
	stack.independent(&X_a[0], n); // Identify independent variables
	stack.dependent(&Y[0], m); // Identify dependent variables
	stack.jacobian(Jac.vec.data()); // Compute & store Jacobian in jac
}


static Vec<double> DrcSol(SMat& SCM, Vec<double>& RHS) {
	MKL_INT NROWS = SCM.rownum;
	MKL_INT NCOLS = SCM.colnum;
	MKL_INT NNONZEROS = SCM.elnum;
	MKL_INT NRHS = 1;
	_MKL_DSS_HANDLE_t handle;
	_INTEGER_t error;
	_CHARACTER_t statIn[] = "determinant", *uplo;
	_DOUBLE_PRECISION_t statOut[5], eps = 1e-6;
	Vec<double> solvec(NROWS);
	solvec.SetValSca(0.0);
	MKL_INT opt = MKL_DSS_DEFAULTS;
	MKL_INT sym = MKL_DSS_NON_SYMMETRIC;
	MKL_INT type = MKL_DSS_INDEFINITE;
	error = dss_create(handle, opt);
	if (error != 0)
		goto PRTERROR;
	error = dss_define_structure(handle, sym, SCM.rowind.data(), NROWS, NCOLS, SCM.colind.data(), NNONZEROS);
	if (error != 0)
		goto PRTERROR;
	error = dss_reorder(handle, opt, 0);
	if (error != 0)
		goto PRTERROR;
	error = dss_factor_real(handle, opt, SCM.data.data());
	if (error != 0)
		goto PRTERROR;
	error = dss_solve_real(handle, opt, RHS.vec.data(), NRHS, solvec.vec.data());
	if (error != 0)
		goto PRTERROR;
	else
		goto SUCCES;
PRTERROR: cout << "ERROR HAS OCCURED IN DIRECT SPARSE SOLVER WITH CODE " + error << endl;
SUCCES: cout << "SOLUTION IS FOUND WITH SUCCESS WITH DIRECT SPARSE SOLVER" << endl;
	error = dss_delete(handle, opt);
	return solvec;
}


static Vec<double> GMRES_NPc(SMat& SCM, Vec<double>& RHS, Vec<double>& Init) {
	MKL_INT N = SCM.colnum;
	MKL_INT size = 128;
	MKL_INT ivar = N;
	MKL_INT itercount;
	MKL_INT RCI_request;
	vector<int> ipar(size);
	vector<double> dpar(size, 0.0);
	vector<double> tmp(N * (2 * N + 1) + (N * (N + 9)) / 2 + 1, 0.0);
	Vec<double> computed_solution = Init;
	dfgmres_init(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data());
	cout << RCI_request << endl;
	ipar[8] = 1;
	ipar[9] = 0;
	ipar[11] = 1;
	dpar[0] = 1.0E-6;
	dfgmres_check(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data());
	cout << RCI_request << endl;
ITER:dfgmres(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data());
	cout << RCI_request << endl;
	if (RCI_request == 1)
	{
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, SCM.BLSmtrx, SCM.descrBLSmtrx, &tmp[ipar[21] - 1], 0.0, &tmp[ipar[22] - 1]);
		goto ITER;
	}
	else {
		goto FAILED;
	}
FAILED:
	cout << "GMRES HAS FAILED" << endl;
COMPLETE:dfgmres_get(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data(), &itercount);
	cout << "GMRES WITHOUT PC HAS SUCCEEDED" << endl;
	MKL_Free_Buffers();
	return computed_solution;
}


static Vec<double> GMRES_Pc_ILU0(SMat& SCM, Vec<double>& RHS, Vec<double>& Init) {
	MKL_INT N = SCM.colnum;
	MKL_INT size = 128;
	MKL_INT matsize = SCM.elnum, incx = 1;
	MKL_INT ivar = N;
	MKL_INT ierr = 0;
	MKL_INT itercount;
	MKL_INT RCI_request;
	MKL_INT maxfil;
	MKL_INT I;
	double tol, dvar;
	double ref_norm2 = 7.772387E+0, nrm2;
	sparse_matrix_t csrL;
	struct matrix_descr descrL;
	vector<int> ipar(size);
	vector<double> dpar(size, 0.0);
	vector<double> residual(size, 0.0);
	vector<double> tmp(N * (2 * N + 1) + (N * (N + 9)) / 2 + 1, 0.0);
	vector<double> bilu0(SCM.elnum, 0.0);
	vector<double> b(N, 0.0);
	vector<double> trvec(N, 0.0);
	Vec<double> computed_solution = Init;
	dfgmres_init(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data());
	ipar[30] = 1;
	dpar[30] = 1.E-20;
	dpar[31] = 1.E-16;
	dcsrilu0(&ivar, SCM.data.data(), SCM.rowind.data(), SCM.colind.data(), bilu0.data(), ipar.data(), dpar.data(), &ierr);
	nrm2 = dnrm2(&matsize, bilu0.data(), &incx);
	mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ONE, N, N, SCM.rowind.data(), SCM.rowind.data() + 1, SCM.colind.data(), bilu0.data());
	printf("Preconditioner dcsrilu0 has returned the ERROR code %d\n", ierr);
	ipar[14] = 2;
	ipar[7] = 0;
	ipar[10] = 1;
	dpar[0] = 1.0E-5;
	dfgmres_check(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data());
	cout << RCI_request << endl;
ITER:dfgmres(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data());
	cout << RCI_request << endl;
	if (RCI_request == 0)
		goto COMPLETE;
	if (RCI_request == 1)
	{
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, SCM.BLSmtrx, SCM.descrBLSmtrx, &tmp[ipar[21] - 1], 0.0, &tmp[ipar[22] - 1]);
		goto ITER;
	}
	if (RCI_request == 2)
	{
		ipar[12] = 1;
		dfgmres_get(&ivar, computed_solution.vec.data(), b.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data(), &itercount);
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, SCM.BLSmtrx, SCM.descrBLSmtrx, b.data(), 0.0, residual.data());
		dvar = -1.0E0;
		I = 1;
		daxpy(&ivar, &dvar, RHS.vec.data(), &I, residual.data(), &I);
		dvar = dnrm2(&ivar, residual.data(), &I);
		if (dvar < 1.0E-5)
			goto COMPLETE;
		else
			goto ITER;
	}
	if (RCI_request == 3)
	{
		descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
		descrL.mode = SPARSE_FILL_MODE_LOWER;
		descrL.diag = SPARSE_DIAG_UNIT;
		mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrL, descrL, &tmp[ipar[21] - 1], trvec.data());
		descrL.mode = SPARSE_FILL_MODE_UPPER;
		descrL.diag = SPARSE_DIAG_NON_UNIT;
		mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrL, descrL, trvec.data(), &tmp[ipar[22] - 1]);
		goto ITER;
	}
	if (RCI_request == 4)
	{
		if (dpar[6] < 1.0E-12)
			goto COMPLETE;
		else
			goto ITER;
	}
COMPLETE:ipar[12] = 0;
	dfgmres_get(&ivar, computed_solution.vec.data(), RHS.vec.data(), &RCI_request, ipar.data(), dpar.data(), tmp.data(), &itercount);
	cout << "GMRES WITH ILU0 PC HAS SUCCEEDED" << endl;
	MKL_Free_Buffers();
	mkl_sparse_destroy(csrL);
	return computed_solution;
}
/*******************************************************************************/