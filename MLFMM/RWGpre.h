#ifndef RWGPRE_H
#define RWGPRE_H
//#ifndef EIGEN_USE_MKL_ALL
//#define EIGEN_USE_MKL_ALL
//#endif
#ifndef EIGEN_DENSE_H
#define EIGEN_DENSE_H
#include <Eigen/Dense>
#endif
#include <fstream>
#include <iostream>
#include <iomanip>

const int GLPN = 3;//��˹���õµ���
const int GLPN2 = GLPN * 2;

const float GL_points[GLPN][4] = {
	0.3333333333333333, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667,
	0.3333333333333333, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
	0.3333333333333333, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
};

struct RWG
{
	std::vector<Eigen::Vector3f> points;
	std::vector<Eigen::Vector3i> triangles;
	std::vector<Eigen::Vector2i> edges;
	std::vector<float> edges_length;
	std::vector<Eigen::Matrix<float, 3, 6> > J_GL;
	std::vector<Eigen::Vector2i> vertex_edges;//�ߵĶ�������

	std::vector<Eigen::Vector3f> tri_normal;//�������ⷨ����
	std::vector<Eigen::Matrix<float, 3, 3>> tri_l;//�����αߵ�λ����
	std::vector<Eigen::Matrix<float, 3, 3>> tri_l_normal;//�����α��ⷨ����
	std::vector<float> tri_area;//���������

	RWG& ReadNas(std::string filename, int triangle_num);
	RWG& initial();
	std::complex<float> GetZij(int i, int j);
	Eigen::Vector3f start_point{ 1e9,1e9,1e9 }, end_point{ -1e9,-1e9,-1e9 };
	Eigen::Vector3f GetEdgeCenter(int index);

private:
	
	float TriDistence(int i, int j);
	
	Eigen::Vector4f GetI1I2(Eigen::Vector3f r, int tri_index);
	Eigen::Vector2i GetEdgeVertex(int edge_index);
};



#endif //RWGPRE_H