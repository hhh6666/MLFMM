#ifndef BASE_H
#define BASE_H

#include <complex>
#include <cmath>
#include <vector>
#ifndef EIGEN_DENSE_H
#define EIGEN_DENSE_H
#include <Eigen/Dense>
#endif

//typedef double JD;
typedef float JD;

typedef std::complex<JD> CP;
typedef Eigen::Matrix<CP, Eigen::Dynamic, Eigen::Dynamic> MatCP;
typedef Eigen::Matrix<JD, Eigen::Dynamic, Eigen::Dynamic> MatJD;
typedef Eigen::Vector<JD, Eigen::Dynamic> VecJD;
typedef Eigen::Vector<CP, Eigen::Dynamic> VecCP;
typedef Eigen::Vector<JD, 3> VecJD3;
typedef Eigen::Vector<CP, 3> VecCP3;



const CP cpdj = CP(0, 1.0);
const JD pi = 3.1415926535897931159979634685441851615906;
const JD c0 = 299792458.0;
const JD mu0 = JD(4.0) * pi * JD(1e-7);
const JD epsilon0 = 8.854187817e-12;
const JD rad = pi / JD(180.0);
const JD Zf = sqrt(mu0 / epsilon0);
const JD min_eps = std::numeric_limits<JD>::epsilon() * JD(2.0);

inline int GetLH(const JD D, const JD k) { return k * D / JD(2.0) + JD(1.8) * pow(log(JD(1.0 / 0.01)), JD(2.0) / JD(3.0)) * pow(k * D / JD(2.0), 1.0 / 3.0); }
inline int GetL(const JD D, const JD k) {return k * D + JD(1.8) * pow(log(JD(1.0 / 0.01)), JD(2.0) / JD(3.0)) * pow(k * D, JD(1.0) / JD(3.0));}
inline int GetN(const JD D, const JD k) { return 2 * int(k * D * 0.5 + 10) + 2; }
inline JD MemUsed(JD mem) { return mem * sizeof(mem) / 1024.0 / 1024.0 / 1024.0; }


inline CP Get_hankel2(int n, JD kd, CP hn_1, CP hn_2)
{
	if (n == 0) return exp(-cpdj * kd) / (-cpdj * kd);
	else if (n == 1) return -(JD(1.0) - cpdj / kd) * exp(-cpdj * kd) / kd;
	else return (JD(2.0) * JD(n) - JD(1.0)) * hn_1 / kd - hn_2;
}
inline JD GetLegendre(int n, JD x, JD pn_1, JD pn_2)
{
	if (n == 0) return JD(1.0);
	else if (n == 1) return x;
	else return ((JD(2.0) * JD(n) - JD(1.0)) * x * pn_1 - (JD(n) - JD(1.0)) * pn_2) / JD(n);
}
inline CP TranferF(int L, JD ct, JD kD, JD wavenumber)
{
	CP TL(0, 0);
	CP hn_1 = -(JD(1.0) - cpdj * kD) * exp(-cpdj * kD) / kD;
	CP hn_2 = exp(-cpdj * kD) / kD;
	JD pn_2 = 1.0, pn_1 = ct;
	for (int l = 0; l < L; ++l) {
		//CPD hankel = std::sph_bessel(l, kD) - cpdj * std::sph_neumann(l, kD);
		CP hankel = Get_hankel2(l, kD, hn_1, hn_2);
		JD pn = GetLegendre(l, ct, pn_1, pn_2);
		hn_2 = hn_1;
		hn_1 = hankel;
		pn_2 = pn_1;
		pn_1 = pn;
		TL += CP(pow(-cpdj, l)) * (JD(2.0 * l) + JD(1.0)) * hankel * pn;
	}
	TL *= (-cpdj * wavenumber) / (JD(4.0) * pi);
	return TL;
}

inline CP LagrangeITF(std::vector<CP>& TArray, JD kdotD, JD diff_theta, int p)
{
	CP TL_theta = CP(0, 0);
	int M = TArray.size();
	JD theta = acos(kdotD);
	int m0 = theta / diff_theta;
	//std::cout << m0 << std::endl;
	for (int i = m0 - p + 1; i <= m0 + p; ++i) {
		//分三种情况，利用对称性取TArray(m)
		int m = i;
		if (m < 0) m = -m;
		if (m >= M)m = M - (m - M + 2);
		JD Lagrange_coe = 1.0;
		JD mdtheta = JD(i) * diff_theta;
		for (int j = m0 - p + 1; j <= m0 + p; ++j) {
			if (j == i)continue;
			JD jdtheta = JD(j) * diff_theta;
			Lagrange_coe *= (jdtheta - theta) / (jdtheta - mdtheta);
		}
		TL_theta += TArray[m] * Lagrange_coe;
	}
	return TL_theta;
}

class LegendrePolynomial {
public:
	LegendrePolynomial(const int N) : N(N), _r(N + 1), _w(N + 1) {
		// Solve roots and weights
		for (int i = 0; i <= N; ++i) {
			JD dr = 1.0;

			// Find zero
			Evaluation eval(cos(pi * (JD(i) - JD(0.25)) / (JD(N) + JD(0.5))), N);
			do {
				dr = eval.v() / eval.d();
				eval.evaluate(eval.x() - dr);
			} while (fabs(dr) > min_eps);

			this->_r[i] = eval.x();// *(pi) * 0.5 + pi / 2.0;
			this->_w[i] = JD(2.0) / ((JD(1.0) - eval.x() * eval.x()) * eval.d() * eval.d());
		}
	}

	JD root(int i) const { return this->_r[i]; }
	JD weight(int i) const { return this->_w[i]; }
private:
	const int N;
	std::vector<JD> _r;
	std::vector<JD> _w;

	/*! Evaluate the value *and* derivative of the
	*   Legendre polynomial
	*/
	class Evaluation {
	public:
		explicit Evaluation(JD x, const int N) : _x(x), _v(1.0), _d(0), N(N) {
			this->evaluate(x);
		}

		void evaluate(JD x) {
			this->_x = x;

			JD vsub1 = x;
			JD vsub2 = 1.0;
			JD f = JD(1.0) / (x * x - JD(1.0));

			for (int i = 2; i <= N; ++i) {
				this->_v = ((JD(2.0) * JD(i) - JD(1.0)) * x * vsub1 - (JD(i) - JD(1.0)) * vsub2) / JD(i);
				this->_d = JD(i) * f * (x * this->_v - vsub1);

				vsub2 = vsub1;
				vsub1 = this->_v;
			}
		}

		JD v() const { return this->_v; }
		JD d() const { return this->_d; }
		JD x() const { return this->_x; }

	private:
		JD _x;
		JD _v;
		JD _d;
		const int N;
	};
};

struct Parameters
{
	JD freq;
	JD lam;
	JD k0;
	Parameters() = default;
	Parameters(JD freq) :freq(freq), lam(c0 / freq), k0(2 * pi / lam) {}
	void set_freq(JD freq) { this->freq = freq; lam = c0 / freq; k0 = 2 * pi / lam; }
};

const Parameters GlobalParams(5e9);

#endif