#ifndef RWGPRE_H
#define RWGPRE_H
//#ifndef EIGEN_USE_MKL_ALL
//#define EIGEN_USE_MKL_ALL
//#endif

#include <fstream>
#include <iostream>
#include <iomanip>
#include "base.h"

const int GLPN = 3;//高斯勒让德点数
const int GLPN2 = GLPN * 2;

const JD GL_points[GLPN][4] = {
	0.3333333333333333, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667,
	0.3333333333333333, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
	0.3333333333333333, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
};

struct RWG
{
	std::vector<VecJD3> points;
	std::vector<Eigen::Vector3i> triangles;
	std::vector<Eigen::Vector2i> edges;
	std::vector<JD> edges_length;
	std::vector<Eigen::Matrix<JD, 3, 6> > J_GL;
	std::vector<Eigen::Vector2i> vertex_edges;//边的顶点索引

	std::vector<VecJD3> tri_normal;//三角形外法向量
	std::vector<Eigen::Matrix<JD, 3, 3>> tri_l;//三角形边单位向量
	std::vector<Eigen::Matrix<JD, 3, 3>> tri_l_normal;//三角形边外法向量
	std::vector<JD> tri_area;//三角形面积

	RWG& ReadNas(std::string filename, int triangle_num);
	RWG& initial();
	CP GetZij(int i, int j);
	VecJD3 start_point{ 1e9,1e9,1e9 }, end_point{ -1e9,-1e9,-1e9 };
	const VecJD3 GetEdgeCenter(int index) const;

private:
	JD TriDistence(int i, int j);
	const JD eps = 0.03;
	Eigen::Vector<JD, 4> GetI1I2(const VecJD3& r, int tri_index);
};

inline void GMRES(const Eigen::VectorXcd& b, const Eigen::MatrixXcd& Z, Eigen::VectorXcd& J)
{
	Eigen::VectorXcd Zb = b;
	clock_t start, end;
	start = clock();
	double Zb_norm = Zb.norm();
	size_t restart_num = 1;
	double iterations_max = 100;
	double lieshu = J.rows();
	double threshold = 1e-3;
	for (int j = 0; j < restart_num; ++j) {
		Eigen::VectorXcd r;
		r = Zb;
		Eigen::VectorXcd c = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::VectorXcd s = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::VectorXcd e1 = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::VectorXcd beta = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(iterations_max, iterations_max);
		Eigen::MatrixXcd V = Eigen::MatrixXcd::Zero(lieshu, iterations_max);
		e1[0] = CP(1.0, 0);
		double r_norm = r.norm();
		double error = r_norm / Zb_norm;
		int number = iterations_max - 2;
		V.col(0) = r / r_norm;
		beta = r_norm * e1;
		for (size_t k = 0; k < iterations_max - 1; ++k) {
			V.col(k + 1).noalias() = (Z * V.col(k));

			for (size_t i = 0; i <= k; ++i) {
				H(i, k) = V.col(i).dot(V.col(k + 1));
				V.col(k + 1).noalias() -= H(i, k) * V.col(i);
			}
			H(k + 1, k) = V.col(k + 1).norm();

			/*if (abs(H(k + 1, k)) < 1e-10) {
				number = k;
				break;
			}*/
			V.col(k + 1) /= H(k + 1, k);
			
			for (size_t i = 0; i < k; ++i) {
				CP temp = c[i] * H(i, k) + s[i] * H(i + 1, k);
				//H(i + 1, k) = -s[i] * H(i, k) + c[i] * H(i + 1, k);
				H(i + 1, k) = -conj(s[i]) * H(i, k) + conj(c[i]) * H(i + 1, k);
				H(i, k) = temp;
			}
			double tao = sqrt(norm(H(k, k)) + norm(H(k + 1, k)));

			c[k] = abs(H(k, k)) * exp(cpdj * arg(H(k + 1, k))) / tao;
			s[k] = abs(H(k + 1, k)) * exp(cpdj * arg(H(k, k))) / tao;
			
			H(k, k) = c[k] * H(k, k) + s[k] * H(k + 1, k);
			H(k + 1, k) = CP(0, 0);
			
			beta[k + 1] = -conj(s[k]) * beta[k];
			beta[k] = c[k] * beta[k];

			error = abs(beta[k + 1]) / Zb_norm;
			std::cout << k << "误差 " << error << std::endl;
			//end = clock();
			//std::cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
			if (error < threshold) {
				number = k;
				break;
			}
		}

		++number;
		Eigen::MatrixXcd h = H.block(0, 0, number, number);
		Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXcd> > lu(h);
		Eigen::VectorXcd y = lu.solve(beta.topRows(number));
		J.noalias() += V.leftCols(number) * y;
		std::cout << "实际误差" << number << " " << (b - Z * J).norm() / b.norm() << " " << error << std::endl;
		if (error < threshold) {
			break;
		}
	}
}

class MOM
{
	RWG& rwg;
	Eigen::MatrixXcd Z;
	Eigen::VectorXcd x;
	Eigen::VectorXcd b;
	Eigen::VectorXcd b_temp;
	std::vector<Eigen::PartialPivLU<MatCP >> lu_list;
	std::vector<int> Z_index;
	const int N = 20;
	void FillZ();
	void Fillb(double theta, double phi);
	void Gmres(VecCP& x, VecCP& b);
	void Prod(const VecCP& x, CP* b_str);
	void SelfProd(VecCP& b, CP* b_str);
public:
	MOM(RWG& rwg) :rwg(rwg) {
		this->FillZ();
		std::cout << "Z填充完成" << std::endl;
		//std::cout << Z.block(0, 0, 4, 4) << std::endl;
		this->Fillb(90, 0);
		b_temp.resize(b.size());
		std::cout << "b填充完成" << " "<<b.squaredNorm()<<std::endl;
		std::cout << "矩阵范数" << Z.squaredNorm() << std::endl;
		x = Eigen::VectorXcd::Zero(Z.rows());
		//x = Z.inverse() * b; 
		/*Eigen::PartialPivLU<Eigen::MatrixXcd > lu(Z);
		x = lu.solve(b);
		VecCP b_hh(x.size()), b_pre(Z.rows());
		std::cout << "？" << std::endl;
		Prod(x, b_hh.data());
		std::cout << "？" << std::endl;
		SelfProd(b, b_pre.data());
		std::cout << "误差：" << (b_pre - b_hh).norm() / b_pre.norm() << " " << x.squaredNorm() << std::endl;*/
		Gmres(x, b);
		//GMRES(b, Z, x);
		std::cout << "误差：" << (Z * x - b).norm() / b.norm() << " " << x.squaredNorm() << std::endl;
	};

	Eigen::Vector2cd FarField(double theta, double phi);
};

inline void mom_ceshi()
{
	RWG rwg;
	rwg.ReadNas("feko/sphere/1094.nas", 1094).initial();
	MOM mom(rwg);
	double dif = 1.0;
	int num = (180 + 1e-10) / dif;
	for (int i = 0; i <= num; ++i) {
		double phi = double(i) * dif;
		Eigen::Vector2cd E = mom.FarField(90.0, phi);
		std::cout << phi << "  " << abs(E[0]) << "    " << arg(E[0]) / rad << "   " << 20 * log10(abs(E[0])) << std::endl;
	}
	/*for (int i = 0; i <= num * 2; ++i) {
		double theta = -180.0 + double(i) * dif;
		Eigen::Vector2cd E = mom.FarField(theta, 0.0);
		std::cout << theta << "    " << arg(E[0]) / rad << "   " << abs(E[0]) << std::endl;
	}*/
}

inline void GMRES_test()
{
	int n = 100;
	MatCP A = MatCP::Random(n, n);
	VecCP b = VecCP::Random(n);
	VecCP x = VecCP::Zero(n);
	GMRES(b, A, x);
}


#endif //RWGPRE_H