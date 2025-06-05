#ifndef SPECTRUMPRE_H
#define SPECTRUMPRE_H

#ifndef EIGEN_DENSE_H
#define EIGEN_DENSE_H
#include <Eigen/Dense>
#endif
#ifndef EIGEN_SPARSE_H
#define EIGEN_SPARSE_H
#include <Eigen/Sparse>
#endif
#include"MPIpre.h"
#include"base.h"

typedef Eigen::Triplet<JD> T_;

inline int GetThetaNum(int L) { return L + 1; }
inline int GetPhiNum(int L) { return 2 * L + 2; }
inline VecJD3 GetK(JD theta, JD phi, JD wavenumber) {
	JD ct = cos(theta);
	JD st = sin(theta);
	JD cp = cos(phi);
	JD sp = sin(phi);
	return wavenumber * VecJD3{ st * cp, st * sp, ct };
}

class TWIP
{
	std::vector<JD>& fa_thetas;
	std::vector<JD>& son_thetas;
	int fa_phi_num, son_phi_num;
	
	int GetThetaIndex(int theta_index) {
		if (theta_index < 0) return -theta_index - 1;
		else if (theta_index >= son_thetas.size()) return 2 * son_thetas.size() - theta_index - 1;
		else return theta_index;
	}
	int GetIndex(int theta_index, int phi_index) {
		int phis_half = son_phi_num / 2;
		if (theta_index < 0 || theta_index >= son_thetas.size()) {
			if (phi_index < phis_half) phi_index += phis_half;
			else if (phi_index >= phis_half) phi_index -= phis_half;
		}
		return GetThetaIndex(theta_index) * son_phi_num + phi_index;

	}
	JD IsVirtual(int theta_index) {
		if (theta_index < 0 || theta_index >= son_thetas.size()) return -1.0;
		return 1.0;
	}
	JD GetTheta(int theta_index) {
		if (theta_index < 0) return -son_thetas[GetThetaIndex(theta_index)];
		else if (theta_index >= son_thetas.size()) return 2.0 * pi - son_thetas[GetThetaIndex(theta_index)];
		else return son_thetas[theta_index];
	}
public:
	static const int p = 3;
	static const int pp = 5;
	TWIP(std::vector<JD>& fa_thetas, std::vector<JD>& son_thetas, int fa_phi_num, int son_phi_num) :
		fa_thetas(fa_thetas), son_thetas(son_thetas), fa_phi_num(fa_phi_num), son_phi_num(son_phi_num) {}
	void GetMatrix(int fa_theta_st,int fa_theta_end,int son_theta_st,int son_theta_end);
	Eigen::SparseMatrix<JD, Eigen::RowMajor> First, Second;
};

class SpectrumPre
{
	const MPIpre& mpipre;
	void Get_Spectrum();
public:
	SpectrumPre() = default;
	SpectrumPre(int level_num, const std::vector<int>& gg, const MPIpre& mpipre)
		:level_num(level_num), actual_L(gg), mpipre(mpipre)
	{
		this->Get_Spectrum();
	}
	std::vector<Eigen::Matrix<JD, 2, 3>> TCtoS;
	std::vector<std::vector<VecJD3>> k_vecs;
	std::vector<std::vector<JD> > thetas;
	std::vector<std::vector<JD>> weights;
	const std::vector<int>& actual_L;
	const int level_num;
	const int GetTopLevelNum() const { return level_num - 2; }
	std::vector<int> thetas_num, phis_num;
	std::vector<int> sptm_thetas_end;
	std::vector<int> sptm_thetas_st;
	std::vector<int> ip_thetas_end;
	std::vector<int> ip_thetas_st;
	const int GetSptmNumLevel(int level) {return weights[level].size(); }

};

#endif