#pragma once

#include"Octree.h"
#include"SpectrumPre.h"
#include"base.h"
#include<queue>

class MatrixPre
{
	
	MPIpre& mpipre;
	std::vector<MatCP> aggregations;
	Eigen::SparseMatrix<JD, Eigen::RowMajor> F, S;
	Eigen::SparseMatrix<JD, Eigen::ColMajor> aF, aS;
	std::vector<Eigen::SparseMatrix<JD, Eigen::RowMajor> > Fu_list, Fm_list, Fd_list, S_list;
	std::vector<Eigen::SparseMatrix<JD, Eigen::ColMajor> > aF_list, aSu_list, aSm_list, aSd_list;
	std::vector<std::vector<VecCP> > phase_shifts;
	std::vector<std::vector<VecCP> > transfers;
	CP* mem_st_ptr = nullptr, * mem_end_ptr = nullptr;
	CP* mpi_mem_st_ptr = nullptr, * mpi_mem_end_ptr = nullptr;
	CP* temp_mem_ptr = nullptr;
	std::vector<MatCP> near_matrix_list;
	std::vector<Eigen::PartialPivLU<MatCP >> Zinv_list;
	const int send_buffer_size = 1;
	const int recv_buffer_size = 1;
	void GetTransfers();
	void MemPre();
	void GetNearNb();
	void GetAggregations();
	void GetInterpolations();
	void GetPhaseShifts();
	void StartTransfer(const int level);
	inline void hhc(int level, CP* sptm_ptr) {
		VecCP e = VecCP3::Zero();
		auto& thetas = spectrum_pre.thetas[level];
		int phi_num = spectrum_pre.phis_num[level];
		int sptm_num = thetas.size() * phi_num;
		Eigen::Map<VecCP> spectrum_theta(sptm_ptr, sptm_num);
		Eigen::Map<VecCP> spectrum_phi(sptm_ptr + sptm_num, sptm_num);
		auto& cube = octree.GetCubesLevel(level)[0];
		JD dif = 2.0 * pi / phi_num;
		for (int k = 0; k < thetas.size(); ++k) {
			for (int g = 0; g < phi_num; g++) {
				int index = k * phi_num + g;
				JD ct = cos(thetas[k]);
				JD st = sin(thetas[k]);
				JD cp = cos(g * dif);
				JD sp = sin(g * dif);
				Eigen::Matrix<JD, 2, 3> T;
				T << ct * cp, ct* sp, -st,
					-sp, cp, 0;
				e += T.row(0).transpose() * spectrum_theta[index] + T.row(1).transpose() * spectrum_phi[index];
			}
		}
		VecJD3 pos = octree.mortoncode3d.GetPoint(cube.mtc, level);
		std::cout << pos.transpose() << " " << CP(0.0, -Zf * GlobalParams.k0 / (4.0 * pi)) * e.transpose() << std::endl;
	}
	

public:
	MatrixPre(OctreeRWG& octree, SpectrumPre& spectrum_pre, MPIpre& mpipre) :
		octree(octree), spectrum_pre(spectrum_pre), mpipre(mpipre)
	{
		this->MemPre();
		this->GetInterpolations();
		this->GetPhaseShifts();
		this->GetAggregations();
		this->GetTransfers();
		this->GetNearNb();
	}
	void AggregationProd(CP* J_ptr, bool judge);
	void NearProd(const VecCP& x, VecCP& b);
	void SelfProd(const VecCP& x, CP* b_str);
	void InterpolationProd(bool judge);
	OctreeRWG& octree;
	SpectrumPre& spectrum_pre;
	const int GetRwgNum() { return octree.rwg.edges.size(); }
	Eigen::Vector2cd GetRwgEFarField(const JD theta, const JD phi, const Eigen::VectorXcd& J);
	Eigen::Vector3cd GetRwgENearField(const VecJD3 r, const Eigen::VectorXcd& J);
	inline std::vector<int> rwg_index_cube(int level) {
		std::set<int> ttt;
		auto& cubes = octree.GetCubesLevel(level);
		auto& proxy_cube1 = octree.GetProxyCubesLevel(level)[0];
		auto& cubes_floor = octree.GetCubesLevel(0);
		for (int i = 0; i < octree.rwg.edges.size(); i++) {
			const VecJD3 point = octree.rwg.GetEdgeCenter(i);
			size_t mtc = (octree.mortoncode3d.GetMortonCode(point));
			for (auto& index : proxy_cube1.local_index) {
				if ((mtc >> (3 * level)) == cubes[index].mtc) {
					int m0 = lower_bound(cubes_floor.begin(), cubes_floor.end(), mtc, [](const Cube& a, size_t b) {
						return a.mtc < b; }) - cubes_floor.begin();
						ttt.insert(m0);
				}
			}
		}
		return std::vector<int>(ttt.begin(), ttt.end());
	}
};
