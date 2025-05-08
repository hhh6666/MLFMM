#ifndef MLFMM_H
#define MLFMM_H
#include "MatrixPre.h"
using std::endl;
using std::cout;
class MLFMM
{
	MatrixPre& matrix_pre;
	MPIpre& mpipre;
	Eigen::VectorXcd J_end;
	Eigen::VectorXcd b_temp;
	
public:
	MLFMM() = default;
	MLFMM(MatrixPre& matrix_pre, MPIpre& mpipre) : matrix_pre(matrix_pre), mpipre(mpipre) {
		b_temp.resize(matrix_pre.octree.b.rows());
	}
	
	void GetFarEfield(double dif)
	{
		/*int num = (180 + 1e-10) / dif;
		for (int i = 0; i <= num; ++i) {
			double phi = double(i) * dif;
			Eigen::Vector2cd E = matrix_pre.GetRwgEFarField(90.0, phi, J_end);
			Eigen::Vector2cd E_all = Eigen::Vector2cd::Zero();
			MPI_Reduce(E.data(), E_all.data(), 2, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
			if (mpipre.GetRank() == 0)
				std::cout << phi << "  " << abs(E_all[0]) << "    " << arg(E_all[0]) / rad << "   " << 20 * log10(abs(E_all[0])) << std::endl;
		}*/

	}
	void ceshi()
	{
		int rwgs_num_row = matrix_pre.octree.cube_rwgs_num[0];
		int size = matrix_pre.GetRwgNum();
		MatCP Z(rwgs_num_row, size);
		Z.setZero();
		for (int i = 0; i < rwgs_num_row; i++) {
			for (int j = rwgs_num_row; j < size; j++) {
				Z(i, j) = matrix_pre.octree.rwg.GetZij(i, j);
			}
		}
		/*VecCP b1(size), b2(size), x(size);
		x.setOnes();
		b_temp = Z * x;
		matrix_pre.SelfProd(b_temp, b1.data());
		b2.setZero();
		Prod(x.data(), b2.data());
		cout << "误差1：" << (b2 - b1).norm() / b2.norm() << endl;*/
		auto& cubes_floor = matrix_pre.octree.GetCubesLevel(0);
		MatCP Z_near(rwgs_num_row, size);
		std::vector<MatCP> Z_fars;
		Z_near.setZero();
		std::vector<int> rwg_index0, rwg_index1, rwg_index2;
		VecJD3 pos1 = matrix_pre.octree.mortoncode3d.GetPoint(matrix_pre.octree.GetProxyCubesLevel(1)[0].mtc, 1);
		VecJD3 pos0 = matrix_pre.octree.mortoncode3d.GetPoint(matrix_pre.octree.GetProxyCubesLevel(0)[0].mtc, 0);
		//VecJD3 pos_hh = pos0 / matrix_pre.spectrum_pre.length;
		auto& near_cube = matrix_pre.octree.near_cubes[0];
		std::set<int> ggg, all;
		for (auto& index : near_cube.local_index) {
			int st = matrix_pre.octree.cube_rwgs_dif[index];
			int ed = matrix_pre.octree.cube_rwgs_num[index] + st;
			cout << "进场：" << index << " " << st << " " << ed << cubes_floor[index].mtc << endl;
			all.insert(index);
			for (int i = 0; i < rwgs_num_row; i++) {
				for (int j = st; j < ed; ++j) {
					Z_near(i, j) = matrix_pre.octree.rwg.GetZij(i, j);
				}
			}
		}
		for (int i = 0; i < matrix_pre.spectrum_pre.GetTopLevelNum(); ++i) {
			std::vector<int> cube_index = matrix_pre.rwg_index_cube(i);
			MatCP Z_far(rwgs_num_row, size);
			Z_far.setZero();
			for (int j = 0; j < rwgs_num_row; j++) {
				for (auto& index : cube_index) {
					int st = matrix_pre.octree.cube_rwgs_dif[index];
					int ed = matrix_pre.octree.cube_rwgs_num[index] + st;
					for (int k = st; k < ed; ++k) Z_far(j, k) = matrix_pre.octree.rwg.GetZij(j, k);
				}
			}
			Z_fars.push_back(Z_far);
		}
		MatCP Z_dif = Z;
		for (int i = 0; i < matrix_pre.spectrum_pre.GetTopLevelNum(); ++i) {
			Z_dif -= Z_fars[i];
		}
		Z_dif -= Z_near;
		cout << "矩阵误差：" << Z_dif.norm() / Z.norm() << endl;
		VecCP x(size);
		x.setOnes();
		cout << "第一个盒子的正确场值：" << pos1.transpose() << " " << matrix_pre.GetRwgENearField(pos1, x).transpose() << endl;
		cout << "第一个盒子的正确场值：" << pos0.transpose() << " " << matrix_pre.GetRwgENearField(pos0, x).transpose() << endl;
		VecCP b1(rwgs_num_row), b2(size);
		
		b2.setZero();

		/*b1 = Z_near * x;
		matrix_pre.NearProd(x, b2);*/

		/*b1 = Z_far0 * x;
		matrix_pre.AggregationProd(x.data(), 0);
		matrix_pre.InterpolationProd(0);
		matrix_pre.AggregationProd(b2.data(), 1);*/

		
		matrix_pre.AggregationProd(x.data(), 0);
		matrix_pre.InterpolationProd(0);
		matrix_pre.InterpolationProd(1);
		matrix_pre.AggregationProd(b2.data(), 1);
		matrix_pre.NearProd(x, b2);
		b1 = Z * x;
		cout << "正确结果：" << endl << b1 << endl << "错误结果：" << endl << b2.segment(0, rwgs_num_row) << endl;
		cout << "误差：" << endl << (b1 - b2.segment(0, rwgs_num_row)).norm()/b1.norm() << endl;

		cout << b2.segment(0, rwgs_num_row).cwiseQuotient(b1) << endl;
		/*MatCP Z_near = Z.block(0, rwgs_num_row, rwgs_num_row, near_index - rwgs_num_row);
		MatCP Z_far = Z.block(0, near_index, rwgs_num_row, size - near_index);
		VecCP b1(size), b2(size), b3(size), b4(size);
		VecCP x(size);
		x.setOnes();
		b1.setZero();
		matrix_pre.NearProd(x, b1);
		b2.segment(0, rwgs_num_row) = Z_near * x.segment(rwgs_num_row, near_index - rwgs_num_row);
		cout << "误差1：" << (b2.segment(0, rwgs_num_row) - b1.segment(0, rwgs_num_row)).norm() / b2.segment(0, rwgs_num_row).norm() << endl;
		b3.segment(0, rwgs_num_row) = Z_far * x.segment(near_index, size - near_index);
		cout << "正确范数：" << b3.segment(0, rwgs_num_row).squaredNorm() << endl;
		b4.setZero();
		matrix_pre.AggregationProd(x.data(), 0);
		matrix_pre.InterpolationProd(0);
		matrix_pre.InterpolationProd(1);
		matrix_pre.AggregationProd(b4.data(), 1);
		VecJD3 pos{ -0.0376649, -0.0376476, -0.037624 };
		cout << "第一个盒子的正确场值：" << matrix_pre.GetRwgENearField(pos, x).transpose() << endl;
		cout << "正确结果：" << endl << b3.segment(0, rwgs_num_row) << endl << "错误结果：" << endl << (b4.segment(0, rwgs_num_row)) << endl;
		cout << "误差2：" << (b3.segment(0, rwgs_num_row) - b4.segment(0, rwgs_num_row)).norm() / b3.segment(0, rwgs_num_row).norm() << endl;*/
		/*Eigen::PartialPivLU<Eigen::MatrixXcd > lu(Z);
		VecCP J = lu.solve(matrix_pre.octree.b);
		std::cout << "误差：" << (Z * J - matrix_pre.octree.b).norm() / matrix_pre.octree.b.norm() << " " << J.squaredNorm() << std::endl;
		{
			JD dif = 1.0;
			int num = (180 + 1e-10) / dif;
			for (int i = 0; i <= num; ++i) {
				double phi = double(i) * dif;
				Eigen::Vector2cd E = matrix_pre.GetRwgEFarField(90.0, phi, J);
				Eigen::Vector2cd E_all = Eigen::Vector2cd::Zero();
				MPI_Reduce(E.data(), E_all.data(), 2, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
				if (mpipre.GetRank() == 0)
					std::cout << phi << "  " << abs(E_all[0]) << "    " << arg(E_all[0]) / rad << "   " << 20 * log10(abs(E_all[0])) << std::endl;
			}
		}
		{
			int N = 5;
			JD R = 0.5 * GlobalParams.lam;
			JD dif = 2 * R / JD(N-1);
			for (int i = 0; i < N; ++i) {
				JD y = -R + i * dif;
				for (int j = 0; j < N; ++j) {
					JD z = -R + j * dif;
					VecJD3 pos{ R + 0.1 * GlobalParams.lam, y, z };
					cout << pos.transpose() << " " << matrix_pre.GetRwgENearField(pos, J).transpose() << endl;
				}
			}
		}*/
		
	}
	void ceshi2() {
		J_end.resize(matrix_pre.GetRwgNum());
		GmresP(matrix_pre.octree.b, J_end);
		std::cout <<" 右端项范数："<< matrix_pre.octree.b.squaredNorm() << std::endl;
		////std::cout << "结束" << std::endl;
		//GetFarEfield(0.25);
		//spectrum_pre.clear_mem();
		JD dif = 0.25;
		int num = (180 + 1e-10) / dif;
		for (int i = 0; i <= num; ++i) {
			JD phi = JD(i) * dif;
			Eigen::Vector2cd E = matrix_pre.GetRwgEFarField(90.0, phi, J_end);
			Eigen::Vector2cd E_all = Eigen::Vector2cd::Zero();
			MPI_Reduce(E.data(), E_all.data(), 2, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
			if (mpipre.GetRank() == 0)
			std::cout << phi << "  " << abs(E_all[0]) << "    " << arg(E_all[0]) / rad << "   " << 20 * log10(abs(E_all[0])) << std::endl;
		}
	}
	void Prod(CP* J_ptr, CP* b_ptr)
	{
		b_temp.setZero();
		Eigen::Map<VecCP> JJ(J_ptr, matrix_pre.GetRwgNum());
		matrix_pre.NearProd(JJ, b_temp);
		//cout << 1 << endl;
		matrix_pre.AggregationProd(J_ptr, 0);
		//cout << 2 << endl;
		matrix_pre.InterpolationProd(0);
		//cout << 3 << endl;
		matrix_pre.InterpolationProd(1);
		//cout << 4 << endl;
		matrix_pre.AggregationProd(b_temp.data(), 1);
		//cout << 5 << endl;
		matrix_pre.SelfProd(b_temp, b_ptr);
		Eigen::Map<VecCP> b_map(b_ptr, JJ.rows());
		b_map += JJ;
	}
	void ceshiP() {
		int size = matrix_pre.GetRwgNum();
		std::vector<Cube>& cube_floor = matrix_pre.octree.GetCubesLevel(0);
		VecCP J(size), b(size), x(size);
		x.setOnes();
		b.setZero();
		b_temp.setZero();
		matrix_pre.NearProd(x, b_temp);
		matrix_pre.AggregationProd(x.data(), 0);
		matrix_pre.InterpolationProd(0);
		matrix_pre.InterpolationProd(1);
		matrix_pre.AggregationProd(b_temp.data(), 1);
		matrix_pre.SelfProd(b_temp, b.data());
		//mpiout(b_temp.squaredNorm(), mpipre);
		int num = matrix_pre.octree.cube_rwgs_num[0];
		/*cout << mpipre.GetRank() << " " << b.topRows(num).squaredNorm() << endl;
		cout << b.topRows(num).transpose() << endl;
		cout << mpipre.GetRank() << " " << num << " " << cube_floor[0].mtc << endl;*/
		b += x;
		/*cout << b.topRows(num).transpose() << endl;
		cout << mpipre.GetRank() << " " << b.topRows(num).squaredNorm() << endl;*/
		JD sss = 0;
		for (int i = 0; i < cube_floor.size(); i++) {
			int st = matrix_pre.octree.cube_rwgs_dif[i];
			Eigen::Map<VecCP> gg(b.data() + st, matrix_pre.octree.cube_rwgs_num[i]);
			//cout << cube_floor[i].mtc << " "<< gg.squaredNorm() << endl;
			if (cube_floor[i].mtc < 32000)sss += gg.squaredNorm();
			//mpiout(gg.squaredNorm(), mpipre);
		}
		//Prod(x.data(), b.data());
		//mpiout(sss, mpipre);
		//mpiout(b.squaredNorm(), mpipre);
		//cout << b.squaredNorm() << " "<<sss << endl;
	}
	void GmresP(const Eigen::VectorXcd& b, Eigen::VectorXcd& J);
};

#endif // !MLFMM_H