#ifndef BASE_H
#define BASE_H

#include <complex>
#include <cmath>
#include <vector>

typedef std::complex<float> CP;


const CP cpdj = CP(0, 1.0);
const float pi = 3.1415926535897931159979634685441851615906;
const float c0 = 299792458.0;
const float mu0 = 4.0 * pi * 1e-7;
const float epsilon0 = 8.854187817e-12;
const float rad = pi / 180.0;
const float Zf = sqrt(mu0 / epsilon0);

inline int GetLH(const float D, const float k) { return k * D / 2 + 1.8 * pow(log(1.0 / 0.01), 2.0 / 3.0) * pow(k * D / 2, 1.0 / 3.0); }
inline int GetL(const float D, const float k) {return k * D + 1.8 * pow(log(1.0 / 0.01), 2.0 / 3.0) * pow(k * D, 1.0 / 3.0);}
inline int GetN(const float D, const float k) { return 2 * int(k * D * 0.5 + 10) + 2; }


inline CP Get_hankel2(int n, float kd, CP hn_1, CP hn_2)
{
	if (n == 0) return exp(-cpdj * kd) / (-cpdj * kd);
	else if (n == 1) return -(1.0f - cpdj / kd) * exp(-cpdj * kd) / kd;
	else return (2.0f * n - 1.0f) * hn_1 / kd - hn_2;
}
inline float GetLegendre(int n, float x, float pn_1, float pn_2)
{
	if (n == 0) return 1.0;
	else if (n == 1) return x;
	else return ((2.0 * n - 1.0) * x * pn_1 - (n - 1.0) * pn_2) / n;
}
inline CP TranferF(int L, float ct, float kD, float wavenumber)
{
	CP TL(0, 0);
	CP hn_1 = -(1.0f - cpdj * kD) * exp(-cpdj * kD) / kD;
	CP hn_2 = exp(-cpdj * kD) / kD;
	float pn_2 = 1.0, pn_1 = ct;
	for (int l = 0; l < L; ++l) {
		//CPD hankel = std::sph_bessel(l, kD) - cpdj * std::sph_neumann(l, kD);
		CP hankel = Get_hankel2(l, kD, hn_1, hn_2);
		float pn = GetLegendre(l, ct, pn_1, pn_2);
		hn_2 = hn_1;
		hn_1 = hankel;
		pn_2 = pn_1;
		pn_1 = pn;
		TL += CP(pow(-cpdj, l)) * (2.0f * l + 1.0f) * hankel * pn;
	}
	TL *= (-cpdj * wavenumber) / (4.0f * pi);
	return TL;
}

inline CP TranferF_old(int L, float ct, float kD, float wavenumber)
{
	CP TL(0, 0);
	for (int l = 0; l < L; ++l) {
		CP hankel = std::sph_bessel(l, kD) - cpdj * std::sph_neumann(l, kD);
		TL += CP(pow(-cpdj, l)) * (2.0f * l + 1.0f) * hankel * std::legendre(l, ct);
	}
	TL *= (-cpdj * wavenumber) / (4.0f * pi);
	return TL;
}

inline CP LagrangeITF(std::vector<CP>& TArray, float kdotD, float diff_theta, int p)
{
	CP TL_theta = 0;
	int M = TArray.size();
	float theta = acos(kdotD);
	int m0 = theta / diff_theta;
	//std::cout << m0 << std::endl;
	for (int i = m0 - p + 1; i <= m0 + p; ++i) {
		//分三种情况，利用对称性取TArray(m)
		int m = i;
		if (m < 0) m = -m;
		if (m >= M)m = M - (m - M + 2);
		float Lagrange_coe = 1;
		float mdtheta = i * diff_theta;
		for (int j = m0 - p + 1; j <= m0 + p; ++j) {
			if (j == i)continue;
			float jdtheta = j * diff_theta;
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
			float dr = 1.0;

			// Find zero
			Evaluation eval(cos(pi * (float(i) - 0.25) / (float(N) + 0.5)), N);
			do {
				dr = eval.v() / eval.d();
				eval.evaluate(eval.x() - dr);
			} while (fabs(dr) > 1e-8);

			this->_r[i] = eval.x();// *(pi) * 0.5 + pi / 2.0;
			this->_w[i] = 2 / ((1 - eval.x() * eval.x()) * eval.d() * eval.d());
		}
	}

	float root(int i) const { return this->_r[i]; }
	float weight(int i) const { return this->_w[i]; }
private:
	const int N;
	std::vector<float> _r;
	std::vector<float> _w;

	/*! Evaluate the value *and* derivative of the
	*   Legendre polynomial
	*/
	class Evaluation {
	public:
		explicit Evaluation(float x, const int N) : _x(x), _v(1.0), _d(0), N(N) {
			this->evaluate(x);
		}

		void evaluate(float x) {
			this->_x = x;

			float vsub1 = x;
			float vsub2 = 1.0;
			float f = 1.0 / (x * x - 1.0);

			for (int i = 2; i <= N; ++i) {
				this->_v = ((2.0 * float(i) - 1.0) * x * vsub1 - (float(i) - 1.0) * vsub2) / float(i);
				this->_d = float(i) * f * (x * this->_v - vsub1);

				vsub2 = vsub1;
				vsub1 = this->_v;
			}
		}

		float v() const { return this->_v; }
		float d() const { return this->_d; }
		float x() const { return this->_x; }

	private:
		float _x;
		float _v;
		float _d;
		const int N;
	};
};



#endif