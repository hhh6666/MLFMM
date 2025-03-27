#pragma once

#include"Octree.h"
#include"SpectrumPre.h"
#include"base.h"

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
	std::vector<std::unordered_map<size_t, int>> transfers_index;
	void GetTransfers();
	
public:
	Interpolation(OctreeRWG& octree, SpectrumPre& spectrum_pre, MPIpre& mpipre) :
		octree(octree), spectrum_pre(spectrum_pre), mpipre(mpipre)
	{
		this->GetInterpolations();
		this->GetPhaseShifts();
	}
	void GetInterpolations();
	void GetPhaseShifts();
	void InterpolationProd(bool judge_hh, bool judge);
};
