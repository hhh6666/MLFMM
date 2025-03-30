#pragma once

#include"Octree.h"
#include"SpectrumPre.h"
#include"base.h"
#include<queue>

class Aggregation
{
	std::vector<Eigen::MatrixXcf> aggregations;
	Eigen::SparseMatrix<float, Eigen::RowMajor> F, S;
	Eigen::SparseMatrix<float, Eigen::ColMajor> aF, aS;
	OctreeRWG& octree;
	SpectrumPre& spectrum_pre;
	MPIpre& mpi_pre;
	void GetAggregations();

public:
	Aggregation(OctreeRWG& octree, SpectrumPre& spectrum_pre, MPIpre& mpi_pre) :
		octree(octree), spectrum_pre(spectrum_pre), mpi_pre(mpi_pre) {
		this->GetAggregations();
	}

	void AggregationProd(Eigen::VectorXcf& x, bool judge);
};

class Interpolation
{
	OctreeRWG& octree;
	SpectrumPre& spectrum_pre;
	MPIpre& mpipre;
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor> > Fu_list, Fm_list, Fd_list, S_list;
	std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > aFu_list, aFm_list, aFd_list, aS_list;
	std::vector<std::vector<Eigen::VectorXcf> > phase_shifts;
	std::vector<std::vector<Eigen::VectorXcf> > transfers;
	std::vector<Eigen::MatrixXcf> NearNb_list;
	CP* mem_st_ptr = nullptr, * mem_end_ptr = nullptr;
	CP* mpi_mem_st_ptr = nullptr, * mpi_mem_end_ptr = nullptr;
	CP* temp_mem_ptr = nullptr;
	std::vector<Eigen::MatrixXcf> near_matrix_list;
	std::vector<Eigen::PartialPivLU<Eigen::MatrixXcf >> Zinv_list;
	const int send_buffer_size = 3;
	const int recv_buffer_size = 2;
	void GetTransfers();
	void MemPre();
	void GetNearNb();

public:
	Interpolation(OctreeRWG& octree, SpectrumPre& spectrum_pre, MPIpre& mpipre) :
		octree(octree), spectrum_pre(spectrum_pre), mpipre(mpipre)
	{
		this->MemPre();
		this->GetInterpolations();
		this->GetPhaseShifts();
		this->GetNearNb();
	}
	void NearProd(Eigen::VectorXcf& x, Eigen::VectorXcf& b);
	void GetInterpolations();
	void GetPhaseShifts();
	void InterpolationProd(bool judge_hh, bool judge);
};
