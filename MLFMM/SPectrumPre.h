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

typedef Eigen::Triplet<float> T_;

inline int GetThetaNum(int L) { return L + 1; }
inline int GetPhiNum(int L) { return (2 * L + 2) % 4 ? (2 * L + 4) : (2 * L + 2); }
inline Eigen::Vector3f GetK(float theta, float phi, float wavenumber) {
	float ct = cos(theta);
	float st = sin(theta);
	float cp = cos(phi);
	float sp = sin(phi);
	return wavenumber * Eigen::Vector3f{ st * cp, st * sp, ct };
}

class TWIP
{
	std::vector<float>& fa_thetas;
	std::vector<float>& son_thetas;
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
	float IsVirtual(int theta_index) {
		if (theta_index < 0 || theta_index >= son_thetas.size()) return -1.0;
		return 1.0;
	}
	float GetTheta(int theta_index) {
		if (theta_index < 0) return -son_thetas[GetThetaIndex(theta_index)];
		else if (theta_index >= son_thetas.size()) return 2.0 * pi - son_thetas[GetThetaIndex(theta_index)];
		else return son_thetas[theta_index];
	}
public:
	static const int p = 2;
	static const int pp = 4;
	TWIP(std::vector<float>& fa_thetas, std::vector<float>& son_thetas, int fa_phi_num, int son_phi_num) :
		fa_thetas(fa_thetas), son_thetas(son_thetas), fa_phi_num(fa_phi_num), son_phi_num(son_phi_num) {}
	void GetMatrix(int fa_theta_st,int fa_theta_end,int son_theta_st,int son_theta_end);
	Eigen::SparseMatrix<float, Eigen::RowMajor> First, Second;
};

class SpectrumPre
{
	std::vector<MPI_Request> requests_up;//4xlevel_num
	std::vector<MPI_Request> requests_down;//4xlevel_num
	const MPIpre& mpipre;
	void Get_Spectrum();
public:
	SpectrumPre() = default;
	SpectrumPre(int level_num, float length, float wavenumber, const MPIpre& mpipre)
		:level_num(level_num), length(length), wavenumber(wavenumber), mpipre(mpipre)
	{
		this->Get_Spectrum();
	}
	std::vector<Eigen::Matrix<float, 2, 3>> TCtoS;
	std::vector<std::vector<Eigen::Vector3f>> k_vecs;
	std::vector<std::vector<float> > thetas;
	std::vector<std::vector<float>> weights;
	float length;
	float wavenumber;
	const int level_num;
	const int GetTopLevelNum() const { return level_num - 2; }
	std::vector<int> thetas_num, phis_num;
	std::vector<int> sptm_thetas_end;
	std::vector<int> sptm_thetas_st;
	std::vector<int> ip_thetas_end;
	std::vector<int> ip_thetas_st;
	const int GetSpectrumNumLevel(int level) {return k_vecs[level].size(); }
	long long unsigned int mem_size = 0;
	CP* mem_st = nullptr;
	CP* mem_end = nullptr;
	CP* mem_available = nullptr;
	CP* mem_change_ava(int size) {
		if ((mem_end - mem_available) < size + 2) mem_available = mem_st;
		mem_available += size;
		return mem_available - size;
	}
	CP* temp_mem = nullptr;
	void clear_mem() {
		if (mem_st != nullptr) delete[] mem_st, mem_st = nullptr;
		if (temp_mem != nullptr) delete[] temp_mem, temp_mem = nullptr;
		mem_size = 0;
		mem_available = nullptr;
		mem_end = nullptr;
	}

};

#endif