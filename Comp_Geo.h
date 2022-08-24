#pragma once
#include <iostream>
#include <stdlib.h>
#include <math.h>
using namespace std;


template<int Dim> struct Point {
	double x[Dim];
	Point(const Point &p) {
		for (int i = 0; i < Dim; i++) x[i] = p.x[i];
	}
	Point& operator= (const Point &p) {
		for (int i = 0; i < Dim; i++) x[i] = p.x[i];
		return *this;
	}
	bool operator== (const Point &p) const {
		for (int i = 0; i < Dim; i++) if (x[i] != p.x[i]) return false;
		return true;
	}
	Point(double x0 = 0.0, double x1 = 0.0, double x2 = 0.0) {
		x[0] = x0;
		if (Dim > 1) x[1] = x1;
		if (Dim > 2) x[2] = x2;
		if (Dim > 3) throw("Exceeded Dimensions!");
	}
};


template<int Dim> double Eucdist(const Point<Dim> &p, const Point<Dim> &q) {
	double dd = 0.0;
	for (int j = 0; j < Dim; j++) dd += (q.x[j] - p.x[j]) * (q.x[j] - p.x[j]);
	return sqrt(dd);
}


template<int Dim> Point<Dim> Vecdist(const Point<Dim> &p, const Point<Dim> &q) {
	Point<Dim> Diff;
	for (int j = 0; j < Dim; j++) Diff.x[j] = q.x[j] - p.x[j];
	return Diff;
}


template<int Dim> struct Box {
	Point<Dim> low, hi;
	Box(){}
	Box(const Point<Dim> &mylo, const Point<Dim> &myhi) : low(mylo), hi(myhi){}
};


template<int Dim> double Boxdist(const Box<Dim> &b, const Point<Dim> &p) {
	double dd = 0.0;
	for (int i = 0; i < Dim; i++) {
		if (p.x[i] < b.low.x[i]) dd += (p.x[i] - b.low.x[i]) * (p.x[i] - b.low.x[i]);
		if (p.x[i] > b.hi.x[i]) dd += (p.x[i] - b.hi.x[i]) * (p.x[i] - b.hi.x[i]);
	}
	return sqrt(dd);
}


template<int Dim> bool IsPinB(const Box<Dim> &b, const Point<Dim> &p) {
	for (int i = 0; i < Dim; i++) {
		if (p.x[i] < b.low.x[i] || p.x[i] > b.hi.x[i]) return false;
	}
	return true;
}


template<int Dim> struct Boxnode : Box<Dim> {
	int mom, dau1, dau2, ptlo, pthi;
	Boxnode(){}
	Boxnode(Point<Dim> mylo, Point<Dim> myhi, int mymom, int myd1,
		int myd2, int myptlo, int mypthi) :
		Box<Dim>(mylo, myhi), mom(mymom), dau1(myd1), dau2(myd2),
		ptlo(myptlo), pthi(mypthi) {}
};


template<int Dim> struct KDtree {
	static const double Big;
	int nboxes, npts;
	vector<Point<Dim>> &ptss;
	Boxnode<Dim> *boxes;
	vector<int> ptindx, rptindx;
	double *coords;
	KDtree(vector<Point<Dim>> &pts);
	~KDtree() { delete[] boxes; }
	double disti(int jpt, int kpt);
	int locate(Point<Dim> pt);
	int locate(int jpt);
	int nearest(int jpt);
	int nearest(Point<Dim> pt);
	int locatenear(Point<Dim> pt, double r, int *list, int nmax);
};


template<int Dim> const double KDtree<Dim>::Big(1.0e99);


static void swap_ptr(int *ptr, int k, int j) {
	double temp = ptr[k];
	ptr[k] = ptr[j];
	ptr[j] = temp;
}


static int selecti(const int k, int *indx, int n, double *arr) {
	int i, ia, ir, j, l, mid;
	double a;
	l = 0;
	ir = n - 1;
	for (;;) {
		if (ir <= l + 1) {
			if (ir == l + 1 && arr[indx[ir]] < arr[indx[l]])
				swap_ptr(indx, l, ir);
			return indx[k];
		}
		else {
			mid = (l + ir) >> 1;
			swap_ptr(indx, mid, l + 1);
			if (arr[indx[l]] > arr[indx[ir]]) swap_ptr(indx, l, ir);
			if (arr[indx[l + 1]] > arr[indx[ir]]) swap_ptr(indx, l + 1, ir);
			if (arr[indx[l]] > arr[indx[l + 1]]) swap_ptr(indx, l, l + 1);
			i = l + 1;
			j = ir;
			ia = indx[l + 1];
			a = arr[ia];
			for (;;) {
				do i++; while (arr[indx[i]] < a);
				do j--; while (arr[indx[j]] > a);
				if (j < i) break;
				swap_ptr(indx, i, j);
			}
			indx[l + 1] = indx[j];
			indx[j] = ia;
			if (j >= k) ir = j - 1;
			if (j <= k) l = i;
		}
	}
}


template<int Dim> KDtree<Dim> ::KDtree(vector<Point<Dim>> &pts) :
	ptss(pts), npts(pts.size()), ptindx(npts), rptindx(npts) {
	int ntmp, m, k, kk, j, nowtask, jbox, np, tmom, tdim, ptlo, pthi;
	int *hp;
	double *cp;
	int taskmom[50], taskdim[50];
	for (k = 0; k < npts; k++) ptindx[k] = k;
	m = 1;
	for (ntmp = npts; ntmp; ntmp >>= 1) {
		m <<= 1;
	}
	nboxes = 2 * npts - (m >> 1);
	if (m < nboxes) nboxes = m;
	nboxes--;
	boxes = new Boxnode<Dim>[nboxes];
	coords = new double[Dim * npts];
	for (j = 0, kk = 0; j < Dim; j++, kk += npts) {
		for (k = 0; k < npts; k++) coords[kk + k] = pts[k].x[j];
	}
	Point<Dim> lo(-Big, -Big), hi(Big, Big);
	boxes[0] = Boxnode<Dim>(lo, hi, 0, 0, 0, 0, npts - 1);
	jbox = 0;
	taskmom[1] = 0;
	taskdim[1] = 0;
	nowtask = 1;
	while (nowtask) {
		tmom = taskmom[nowtask];
		tdim = taskdim[nowtask--];
		ptlo = boxes[tmom].ptlo;
		pthi = boxes[tmom].pthi;
		hp = &ptindx[ptlo];
		cp = &coords[tdim * npts];
		np = pthi - ptlo + 1;
		kk = (np - 1) / 2;
		(void)selecti(kk, hp, np, cp);
		hi = boxes[tmom].hi;
		lo = boxes[tmom].low;
		hi.x[tdim] = lo.x[tdim] = coords[tdim * npts + hp[kk]];
		boxes[++jbox] = Boxnode<Dim>(boxes[tmom].low, hi, tmom, 0, 0, ptlo, ptlo + kk);
		boxes[++jbox] = Boxnode<Dim>(lo, boxes[tmom].hi, tmom, 0, 0, ptlo + kk + 1, pthi);
		boxes[tmom].dau1 = jbox - 1;
		boxes[tmom].dau2 = jbox;
		if (kk > 1) {
			taskmom[++nowtask] = jbox - 1;
			taskdim[nowtask] = (tdim + 1) % Dim;
		}
		if (np - kk > 3) {
			taskmom[++nowtask] = jbox;
			taskdim[nowtask] = (tdim + 1) % Dim;
		}
	}
	for (j = 0; j < npts; j++) rptindx[ptindx[j]] = j;
	delete[] coords;
}


template<int Dim> double KDtree<Dim> ::disti(int jpt, int kpt) {
	if (jpt == kpt) return Big;
	else return Eucdist(ptss[jpt], ptss[kpt]);
}


template<int Dim> int KDtree<Dim> ::locate(Point <Dim> pt) {
	int nb, d1, jdim;
	nb = jdim = 0;
	while (boxes[nb].dau1) {
		d1 = boxes[nb].dau1;
		if (pt.x[jdim] <= boxes[d1].hi.x[jdim]) nb = d1;
		else nb = boxes[nb].dau2;
		jdim = ++jdim % Dim;
	}
	return nb;
}


template<int Dim> int KDtree<Dim>::locate(int jpt) {
	int nb, d1, jh;
	jh = rptindx[jpt];
	nb = 0;
	while (boxes[nb].dau1) {
		d1 = boxes[nb].dau1;
		if (jh <= boxes[d1].pthi) nb = d1;
		else nb = boxes[nb].dau2;
	}
	return nb;
}


template<int Dim> int KDtree<Dim> :: nearest(Point<Dim> pt) {
	int i, k, nrst, ntask;
	int task[50];
	double dnrst = Big, d;
	k = locate(pt);
	for (i = boxes[k].ptlo; i <= boxes[k].pthi; i++) {
		d = Eucdist(ptss[ptindx[i]], pt);
		if (d < dnrst) {
			nrst = ptindx[i];
			dnrst = d;
		}
	}
	task[1] = 0;
	ntask = 1;
	while (ntask) {
		k = task[ntask--];
		if (Boxdist(boxes[k], pt) < dnrst) {
			if (boxes[k].dau1) {
				task[++ntask] = boxes[k].dau1;
				task[++ntask] = boxes[k].dau2;
			}
			else {
				for (i = boxes[k].ptlo; i <= boxes[k].pthi; i++) {
					d = Eucdist(ptss[ptindx[i]], pt);
					if (d < dnrst) {
						nrst = ptindx[i];
						dnrst = d;
					}
				}
			}
		}
	}
	return nrst;
 }


template<int Dim> int KDtree<Dim> ::nearest(int jpt) {
	int i, k, nrst, ntask;
	int task[50];
	double dnrst = Big, d;
	k = locate(ptss[ptindx[jpt]]);
	for (i = boxes[k].ptlo; i <= boxes[k].pthi; i++) {
		d = disti(ptss[ptindx[i]], ptss[ptindx[jpt]]);
		if (d < dnrst) {
			nrst = ptindx[i];
			dnrst = d;
		}
	}
	task[1] = 0;
	ntask = 1;
	while (ntask) {
		k = task[ntask--];
		if (disti(boxes[k], ptss[ptindx[jpt]]) < dnrst) {
			if (boxes[k].dau1) {
				task[++ntask] = boxes[k].dau1;
				task[++ntask] = boxes[k].dau2;
			}
			else {
				for (i = boxes[k].ptlo; i <= boxes[k].pthi; i++) {
					d = disti(ptss[ptindx[i]], ptss[ptindx[jpt]]);
					if (d < dnrst) {
						nrst = ptindx[i];
						dnrst = d;
					}
				}
			}
		}
	}
	return nrst;
}


template<int Dim> int KDtree<Dim> ::locatenear(Point<Dim> pt, double r, int *list, int nmax) {
	int k, i, nb, nbold, nret, ntask, jdim, d1, d2;
	int task[50];
	nb = jdim = nret = 0;
	if (r < 0.0) throw("radius must be nonnegative");
	while (boxes[nb].dau1) {
		nbold = nb;
		d1 = boxes[nb].dau1;
		d2 = boxes[nb].dau2;
		if (pt.x[jdim] + r <= boxes[d1].hi.x[jdim]) nb = d1;
		else if (pt.x[jdim] - r >= boxes[d2].low.x[jdim]) nb = d2;
		jdim = ++jdim % Dim;
		if (nb == nbold) break;
	}
	task[1] = nb;
	ntask = 1;
	while (ntask) {
		k = task[ntask--];
		if (Boxdist(boxes[k], pt) > r) continue;
		if (boxes[k].dau1) {
			task[++ntask] = boxes[k].dau1;
			task[++ntask] = boxes[k].dau2;
		}
		else {
			for (i = boxes[k].ptlo; i <= boxes[k].pthi; i++) {
				if (Eucdist(ptss[ptindx[i]], pt) <= r && nret < nmax)
					list[nret++] = ptindx[i];
				if (nret == nmax) return nmax;
			}
		}
	}
	return nret;
}

