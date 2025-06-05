#ifndef RWGPRE_H
#define RWGPRE_H
//#ifndef EIGEN_USE_MKL_ALL
//#define EIGEN_USE_MKL_ALL
//#endif

#include <fstream>
#include <iostream>
#include <iomanip>
#include "base.h"
using std::cout;
using std::endl;

const int GLPN = 3;//高斯勒让德点数
const int GLPN2 = GLPN * 2;
const int GLS = 4;

const JD GL_points[GLPN][4] = {
	0.3333333333333333, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667,
	0.3333333333333333, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
	0.3333333333333333, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
};
//const JD GL_points[GLPN][4] = {
//	-0.5624999999999998, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
//	0.5208333333333332, 0.6000000000000000, 0.2000000000000000, 0.2000000000000000,
//	0.5208333333333332, 0.2000000000000000, 0.2000000000000000, 0.6000000000000000,
//	0.5208333333333332, 0.2000000000000000, 0.6000000000000000, 0.2000000000000000
//};
//M=4
//const JD GL_singular[GLS][4] = {
//	0.3333333333333333, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667,
//	0.3333333333333333, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
//	0.3333333333333333, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
//};
const JD GL_singular[GLS][4] = {
	-0.5624999999999998, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
	0.5208333333333332, 0.6000000000000000, 0.2000000000000000, 0.2000000000000000,
	0.5208333333333332, 0.2000000000000000, 0.2000000000000000, 0.6000000000000000,
	0.5208333333333332, 0.2000000000000000, 0.6000000000000000, 0.2000000000000000
};
////M=6
//const JD GL_singular[GLS][4] = {
//	0.10995174365532, 0.81684757298046, 0.09157621350977, 0.09157621350977,
//	0.10995174365532, 0.09157621350977, 0.09157621350977, 0.81684757298046,
//	0.10995174365532, 0.09157621350977, 0.81684757298046, 0.09157621350977,
//	0.22338158967801, 0.10810301816807, 0.44594849091597, 0.44594849091597,
//	0.22338158967801, 0.44594849091597, 0.44594849091597, 0.10810301816807,
//	0.22338158967801, 0.44594849091597, 0.10810301816807, 0.44594849091597
//};

//const JD GL_singular[GLS][4] = {
//	0.2250000000000002,	0.3333333333333333,	0.3333333333333333,	0.3333333333333333,
//	0.1323941527885061,	0.0597158717897698,	0.4701420641051151,	0.4701420641051151,
//	0.1323941527885061,	0.4701420641051151,	0.4701420641051151,	0.0597158717897698,
//	0.1323941527885061,	0.4701420641051151,	0.0597158717897698,	0.4701420641051151,
//	0.1259391805448272,	0.7974269853530873,	0.1012865073234563,	0.1012865073234563,
//	0.1259391805448272,	0.1012865073234563,	0.1012865073234563,	0.7974269853530873,
//	0.1259391805448272,	0.1012865073234563,	0.7974269853530873,	0.1012865073234563
//};

struct RWG
{
	std::vector<VecJD3> points;
	std::vector<Eigen::Vector3i> triangles;
	std::vector<Eigen::Vector2i> edges;
	std::vector<JD> edges_length;
	std::vector<Eigen::Matrix<JD, 3, GLPN2> > J_GL;
	std::vector<Eigen::Vector2i> vertex_edges;//边的顶点索引

	std::vector<VecJD3> tri_normal;//三角形外法向量
	std::vector<Eigen::Matrix<JD, 3, 3>> tri_l;//三角形边单位向量
	std::vector<Eigen::Matrix<JD, 3, 3>> tri_l_normal;//三角形边外法向量
	std::vector<JD> tri_area;//三角形面积

	RWG& ReadNas(std::string filename, int triangle_num);
	RWG& initial();
	CP GetZij(int i, int j);
	CP Getbi(int i);
	
	VecJD3 start_point{ 1e9,1e9,1e9 }, end_point{ -1e9,-1e9,-1e9 };
	RWG& set_wave(int theta, int phi) {
		this->theta = theta;
		this->phi = phi;
		JD ct = cos(theta * rad), st = sin(theta * rad), cp = cos(phi * rad), sp = sin(phi * rad);
		k_unit = -VecJD3(st * cp, st * sp, ct);//方向相反
		T << ct * cp, ct* sp, -st,
			-sp, cp, 0;
		Eigen::Vector<JD, 2> polor{ -JD(1.0),JD(0.0) };
		Ep = T.transpose() * polor;
		Hp = k_unit.cross(Ep);
		//cout <<"磁场方向："<< Hp << endl;
		return *this;
	}
	const VecJD3 GetEdgeCenter(int index) const;
	const JD alpha = JD(0.8);

private:
	JD TriDistence(int i, int j);
	const JD eps = GlobalParams.lam * JD(0.22);
	const JD eps2 = 1.0;
	Eigen::Vector<JD, 4> GetI1I2(const VecJD3& r, int tri_index);
	Eigen::Vector<JD, 4> GetI3I4(const VecJD3& r, int tri_index);
	Eigen::Vector<JD, 4> GetI3I4_num(const VecJD3& r, int tri_index) {
		Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
		Eigen::Vector3i& tri = triangles[tri_index];
		for (int p = 0; p < GLS; ++p) {
			VecJD3 rp = GL_singular[p][1] * points[tri[0]] + GL_singular[p][2] * points[tri[1]] + GL_singular[p][3] * points[tri[2]];
			JD d = tri_normal[tri_index].dot(r - points[tri[0]]);
			VecJD3 rho = rp - (r - d * tri_normal[tri_index]);
			JD R3 = pow((r - rp).norm(), 3);
			ans.head(3) += GL_singular[p][0] * rho / R3;
			ans[3] += GL_singular[p][0] * d / R3;
			//cout << GL_singular[p][0] << " ??? " << d << " " << 1.0 / R3 << " " << ans[3] << endl;
		}
		return ans;
	}
	Eigen::Vector<JD, 4> GetI1I2_num(const VecJD3& r, int tri_index) {
		Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
		Eigen::Vector3i& tri = triangles[tri_index];
		for (int p = 0; p < GLS; ++p) {
			VecJD3 rp = GL_singular[p][1] * points[tri[0]] + GL_singular[p][2] * points[tri[1]] + GL_singular[p][3] * points[tri[2]];
			JD d = tri_normal[tri_index].dot(r - points[tri[0]]);
			VecJD3 rho = rp - (r - d * tri_normal[tri_index]);
			JD R = (r - rp).norm();
			ans.head(3) += GL_singular[p][0] * rho / R;
			ans[3] += GL_singular[p][0] * 1.0 / R;
			//cout << GL_singular[p][0] << " ??? " << d << " " << 1.0 / R3 << " " << ans[3] << endl;
		}
		return ans;
	}
	JD theta, phi;
	VecJD3 k_unit;
	Eigen::Matrix<JD, 2, 3> T;
	VecJD3 Ep, Hp;
	
	CP GetEFIEij(int i, int j);
	CP GetMFIEij(int i, int j);
	CP GetEbi(int i);
	CP GetHbi(int i);
};


class MOM
{
	RWG& rwg;
	MatCP Z;
	VecCP x;
	VecCP b;
	VecCP b_temp;
	std::vector<Eigen::PartialPivLU<MatCP >> lu_list;
	std::vector<int> Z_index;
	const int N = 20;
	void FillZ();
	void Fillb();
	void Gmres(VecCP& x, VecCP& b);
	
	void SelfProd(VecCP& b, CP* b_str);
public:
	void Prod(const VecCP& x, CP* b_str);
	void start() {
		this->FillZ();
		std::cout << Z.block(0, 0, 10, 4) << std::endl;
		rwg.set_wave(90, 90);
		this->Fillb();
		x = VecCP::Zero(Z.rows());
		Eigen::PartialPivLU<MatCP > lu(Z);
		x = lu.solve(b);
	}
	MOM(RWG& rwg) :rwg(rwg) {
		
	};
	void GetSelf(const std::vector<int>& index) {
		Z_index = index;
		lu_list.reserve(Z_index.size() - 1);
		cout << rwg.edges.size() << endl;
		this->FillZ();
		/*std::cout << "Z填充完成" << std::endl;
		std::cout << Z.block(0, 0, 10, 4) << std::endl;
		for (int i = 0; i < Z_index.size() - 1;++i) {
			int st = Z_index[i], ed = Z_index[i + 1];
			Eigen::PartialPivLU<MatCP > lu(Z.block(st, st, ed - st, ed - st));
			lu_list.push_back(lu);
		}
		rwg.set_wave(90, 90);
		this->Fillb();
		std::cout << "b填充完成" << " " << b.squaredNorm() << std::endl;
		std::cout << "矩阵范数" << Z.squaredNorm() << std::endl;
		x = VecCP::Zero(Z.rows());

		Gmres(x, b);
		std::cout << "误差：" << (Z * x - b).norm() / b.norm() << " " << x.squaredNorm() << std::endl;*/
	}
	Eigen::Vector<CP, 2> FarField(JD theta, JD phi);
};

inline void mom_ceshi()
{
	RWG rwg;
	//rwg.ReadNas("feko/sphere/1lam/1094.nas", 1094).initial();
	//rwg.ReadNas("feko/sphere/1lam/die.nas", 20).initial();
	//rwg.ReadNas("feko/x47b/2/20684.nas", 20684).initial();
	rwg.ReadNas("feko/x47b/3/7058.nas", 7058).initial();
	//rwg.ReadNas("feko/sphere/4lam/17474.nas", 17474).initial();
	//rwg.ReadNas("feko/ermianjiao/mini/16.nas", 16).initial();
	
	//cout << "??" << endl;
	MOM mom(rwg);
	mom.start();
	JD dif = 0.25;
	int num = (180 + min_eps) / dif;
	for (int i = 0; i <= num; ++i) {
		JD phi = JD(i) * dif;
		Eigen::Vector<CP, 2> e = mom.FarField(90.0, phi);
		std::cout << phi << "  " << abs(e[0]) << "    " << arg(e[0]) / rad << "   " << 20 * log10(abs(e[0])) << std::endl;
	}
	/*for (int i = 0; i <= num * 2; ++i) {
		JD theta = -180.0 + JD(i) * dif;
		Eigen::Vector<CP, 2> E = mom.FarField(theta, 0.0);
		std::cout << theta << "    " << abs(E[1]) << "    " << arg(E[1]) / rad << "   " << 20 * log10(abs(E[1])) << std::endl;
	}*/
}

//class Singularity
//{
//	JD freq = 3e9;
//	JD lam = c0 / freq;
//	JD length = lam * JD(1.0);
//	JD k0 = 2.0 * pi / lam;
//	VecJD3 v[3];
//	VecJD3 l[3] = {
//		VecJD3{ 1, 0, 0 },
//		VecJD3{ -JD(0.5),sqrt(JD(3.0)) * JD(0.5),0 },
//		VecJD3{ -JD(0.5),-sqrt(JD(3.0)) * JD(0.5),0 }
//	};
//	VecJD3 u[3] = {
//		VecJD3{ 0, -1, 0 },
//		VecJD3{ sqrt(JD(3.0)) * JD(0.5),JD(0.5),0 },
//		VecJD3{ -sqrt(JD(3.0)) * JD(0.5),JD(0.5),0 }
//	};
//	const VecJD3 n{ 0,0,-1 };
//	VecJD3 r0 = VecJD3{ 1.0,0,0 }*length;
//	JD area;
//	Eigen::Vector<JD, 4> I3I4_analytic(const VecJD3& r)
//	{
//		Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
//		for (int i = 0; i < 3; ++i) {
//			int next = (i + 1) % 3;
//			JD lp = (v[next] - r).dot(l[i]);
//			JD lm = (v[i] - r).dot(l[i]);
//			JD Rp = (v[next] - r).norm();
//			JD Rm = (v[i] - r).norm();
//			JD d = abs(n.dot(r - v[next]));
//			JD P0 = abs((v[i] - r).dot(u[i]));
//			if (abs(P0) < min_eps) P0 += min_eps * 2;
//			//JD R0 = sqrt(d * d + P0 * P0);
//			JD R02 = d * d + P0 * P0;
//			JD f = log(Rp + lp) - log(Rm + lm);
//			ans.head(3) += -u[i] * f;
//			/*if (triangles[tri_index][i] == 0 && triangles[tri_index][(i + 1) % 3] == 1 || triangles[tri_index][i] == 1 && triangles[tri_index][(i + 1) % 3] == 0) {
//				ans.head(3) -= -tri_l_normal[tri_index].col(i) * f;
//			}*/
//			JD P0u = (v[i] - r).dot(u[i]) / P0;
//			/*if (d < min_eps) {
//				JD beta = lp / (P0 * Rp) - lm / (P0 * Rm);
//				ans[3] += -P0u * beta;
//			}
//			else {
//				JD beta = atan(d * lp / (P0 * Rp)) - atan(d * lm / (P0 * Rm)) + atan(lm / P0) - atan(lp / P0);
//				ans[3] += -P0u * beta / d;
//			}*/
//			JD beta = atan(P0 * lp / (R02 + d * Rp)) - atan(P0 * lm / (R02 + d * Rm));
//
//			ans[3] += P0u * beta;
//		}
//		return ans;
//	}
//	Eigen::Vector<JD, 4> I1I2_analytic(const VecJD3& r)
//	{
//		Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
//		for (int i = 0; i < 3; ++i) {
//			int next = (i + 1) % 3;
//			JD d = abs(n.dot(r - v[next]));
//			JD P0 = abs((v[i] - r).dot(u[i]));
//			if (abs(P0) < min_eps) P0 += min_eps * 2;
//			JD R02 = d * d + P0 * P0;
//			JD R0 = sqrt(R02);
//
//			JD lp = (v[next] - r).dot(l[i]);
//			JD lm = (v[i] - r).dot(l[i]);
//			JD Rp = (v[next] - r).norm();
//			JD Rm = (v[i] - r).norm();
//			JD temp = log((Rp + lp) / (Rm + lm));
//			JD f = isnan(temp) || isinf(temp) ? 0.0 : temp;
//			//JD f = R02 < min_eps * eps2 ? 0.0 : temp;
//			JD beta = atan(P0 * lp / (R02 + d * Rp)) - atan(P0 * lm / (R02 + d * Rm));
//			ans.head(3) += u[i] * (R02 * f + lp * Rp - lm * Rm);
//			JD P0u = (v[i] - r).dot(u[i]) / P0;
//			ans[3] += P0u * (P0 * f - d * beta);
//		}
//		ans.head(3) *= JD(0.5);
//		return ans;
//	}
//	Eigen::Vector<JD, 4> I3I4_num(const VecJD3& r) {
//		Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
//		for (int p = 0; p < GLS; ++p) {
//			VecJD3 rp = GL_singular[p][1] * v[0] + GL_singular[p][2] * v[1] + GL_singular[p][3] * v[2];
//			JD d = n.dot(r - r0);
//			VecJD3 rho = rp - r0;
//			JD R3 = pow((r - rp).norm(), 3);
//			ans.head(3) += GL_singular[p][0] * rho / R3;
//			ans[3] += GL_singular[p][0] * d / R3;
//		}
//		return ans;
//	}
//	Eigen::Vector<JD, 4> I1I2_num(const VecJD3& r) {
//		Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
//		for (int p = 0; p < GLS; ++p) {
//			VecJD3 rp = GL_singular[p][1] * v[0] + GL_singular[p][2] * v[1] + GL_singular[p][3] * v[2];
//			JD d = n.dot(r - r0);
//			VecJD3 rho = rp - r0;
//			JD R = (r - rp).norm();
//			ans.head(3) += GL_singular[p][0] * rho / R;
//			ans[3] += GL_singular[p][0] * 1.0 / R;
//		}
//		return ans;
//	}
//public:
//	Singularity& set_freq(JD freq) {
//		this->freq = freq;
//		this->lam = c0 / freq;
//		this->length = lam * JD(1.0);
//		this->k0 = 2.0 * pi / lam;
//		v[0] = VecJD3{ -JD(0.5),0,0 } *length;
//		v[1] = VecJD3{ JD(0.5),0,0 } *length;
//		v[2] = VecJD3{ 0,sqrt(JD(3.0)) * JD(0.5),0 } *length;
//		area = ((v[1] - v[0]).cross(v[2] - v[1])).norm() * JD(0.5);
//		return *this;
//	}
//	void ceshi() {
//		JD dif = 0.005 * lam;
//		JD dist_st = 0.02 * lam;
//		JD dist_ed = 0.5 * lam;
//		int N = (dist_ed - dist_st) / dif;
//		JD ans_ana = 0, ans_num = 0;
//		for (int i = 0; i <= N; ++i) {
//			JD dist = JD(i) * dif + dist_st;
//			VecJD3 r = r0 + VecJD3{ 0,0,dist };
//			JD d = n.dot(r - r0);
//			Eigen::Vector<JD, 4> a1 = I3I4_analytic(r);
//			Eigen::Vector<JD, 4> a2 = I3I4_num(r);
//			Eigen::Vector<JD, 4> b1 = I1I2_analytic(r);
//			Eigen::Vector<JD, 4> b2 = I1I2_num(r);
//			/*VecJD3 an1 = a1.head(3) + (r0 - v[1]) * a1[3];
//			VecJD3 an2 = a2.head(3) + (r0 - v[1]) * a2[3];*/
//			VecJD3 an1 = a1.head(3) - n * a1[3];
//			VecJD3 an2 = (a2.head(3) - n * a2[3]) * area;
//			VecJD3 bn1 = (b1.head(3) - n * d * b1[3]) * k0 * k0 * JD(0.5);
//			VecJD3 bn2 = (b2.head(3) - n * d * b2[3]) * area * k0 * k0 * JD(0.5);
//			ans_ana = an1.norm();
//			ans_num = an2.norm();
//			cout << dist/lam << "   " << ans_ana << "   " << ans_num << "   " << (ans_ana - ans_num) << "         " << a1.transpose() << "      " << a2.transpose()*area << "  "<<area<<endl;
//		}
//	}
//};
//
//inline void sin_ceshi()
//{
//	Singularity s;
//	s.set_freq(3e9);
//	s.ceshi();
//}

#endif //RWGPRE_H