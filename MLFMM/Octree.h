#ifndef OCTREE_H
#define OCTREE_H

#ifndef EIGEN_DENSE_H
#define EIGEN_DENSE_H
#include <Eigen/Dense>
#endif

#include "MPIpre.h"
#include "RWGpre.h"
#include <set>

struct NearProxyCube
{
	size_t mtc;
	int process_index = -1;//���ڽ��̱��
	std::vector<int> local_index;//���׷��͵����ؽ��̺�ȷ��������ڱ��غ��ӵ���ʼλ��
	std::vector<int> index;//�������������
};

struct ProxyCube
{
	size_t mtc;
	int process_index= -1;//���ڽ��̱��
	//int my_index;//����������������̵�λ��
	std::vector<int> local_index;
	std::vector<int> index;

};

inline size_t Part1By2(size_t code)
{
	size_t x = code & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;
	//cout << bitset<32>(code) << " " << bitset<32>(x) << endl;
	return x;
}

struct Cube
{
	size_t mtc;
	CP* sptmtheta = nullptr;
};

struct CubeWieght
{
	int weight = 0;
	int cube_index;
};

struct MortonCode1D
{
	JD start_point;
	JD end_point;
	JD length;
	int level_num;
	MortonCode1D& set_start_point(JD start_point, JD end_point) { this->start_point = start_point; this->end_point = end_point; return *this; }
	MortonCode1D& set_length(JD length) { this->length = length; return *this; }
	size_t GetMortonCode(const JD& point) const;
};

class MortonCode3D
{
	MortonCode1D mortoncode1d[3];
	
	VecJD3 centre;
	int level_num = 1;
	int get_level_num(JD floor_cube_length, JD cube_length) {
		int level = 0;
		while ((1 << level) * floor_cube_length < cube_length) {
			++level;
		}
		level_num = std::max(level_num, level + 1);
		return level;
	}
public:
	MortonCode3D() = default;
	MortonCode3D(VecJD3 start_point, VecJD3 end_point, JD length) {
		this->centre = (start_point + end_point) / 2;
		int a = get_level_num(length, (end_point - start_point)[0]);
		int b = get_level_num(length, (end_point - start_point)[1]);
		int c = get_level_num(length, (end_point - start_point)[2]);
		mortoncode1d[0].level_num = a;
		mortoncode1d[1].level_num = b;
		mortoncode1d[2].level_num = c;
		start_point = centre - (VecJD3{ JD(1 << a),JD(1 << b),JD(1 << c) } *length) / 2;
		end_point = centre + (VecJD3{ JD(1 << a),JD(1 << b),JD(1 << c) } *length) / 2;
		mortoncode1d[0].set_length(length).set_start_point(start_point[0], end_point[0]);
		mortoncode1d[1].set_length(length).set_start_point(start_point[1], end_point[1]);
		mortoncode1d[2].set_length(length).set_start_point(start_point[2], end_point[2]);
		near_length_max.resize(level_num);
		next_near_length_max.resize(level_num);
		for (int i = 0; i < level_num; ++i) {
			near_length_max[i] = length * (1 << i) * sqrt(3.0) + 1e-6;
			next_near_length_max[i] = 3.0 * length * (1 << i) * sqrt(3.0) + 1e-6;
		}
	}
	std::vector<JD> near_length_max;
	std::vector<JD> next_near_length_max;
	VecJD3 GetPoint(size_t morton_code, int level_index) const;
	VecJD3 GetGap(size_t morton_code_gap, int level_index) const;
	VecJD3 GetGap2(size_t mtc1, size_t mtc2, int level_index) const;
	size_t GetMortonCode(const VecJD3& point) const;
	const int& GetLevelNum() const { return level_num; }
	
};

class OctreeRWG
{
	void Greedy(std::vector<Cube>& cubes_level_old, std::vector<CubeWieght>& cube_weights, std::vector<Cube>& cubes_level_new
		, int level_index);
protected:
	std::vector<std::vector<Cube> > cubes;//���غ���
	std::vector<std::vector<ProxyCube> > proxy_cubes;//������ӣ����ڴν��ڣ�
	
	RWG* old_rwg_ptr = nullptr;
	MPIpre& mpipre;
	void Fillcubes();
	void GetNear();
	void change_rwgs();
	std::vector<size_t> GetFarNeighbor(size_t mtc, std::unordered_map<size_t, int>& childs_num);
	std::vector<size_t> GetNearNeighbor(size_t mtc, std::unordered_map<size_t, int>& childs_num);
	//void cubes_sort();

public:
	OctreeRWG() = default;
	OctreeRWG(RWG* old_rwg_ptr, JD length, MPIpre& mpipre)
		:old_rwg_ptr(old_rwg_ptr), mortoncode3d(old_rwg_ptr->start_point, old_rwg_ptr->end_point, length), mpipre(mpipre)
	{
		this->Fillcubes();
		this->GetNear();
		this->change_rwgs();
		rwg.initial();
	};
	MortonCode3D mortoncode3d;
	std::vector<Cube>& GetCubesLevel(int level_index) { return cubes[level_index]; }
	std::vector<ProxyCube>& GetProxyCubesLevel(int level_index) { return proxy_cubes[level_index]; }
	RWG rwg;
	VecCP b;
	void Fillb(JD theta, JD phi);
	std::vector<int> cube_rwgs_num;
	std::vector<int> cube_rwgs_dif;//���к��ӣ��������غʹ�����ӣ������λ�ã�����������ڱ��غ��Ӻ���
	std::vector<int> near_cube_rwgs_num;//ÿһ���������еĻ���������
	std::vector<int> near_cubes_process_index;//��¼ÿһ�����̶�Ӧ�Ľ�������ӵ���ʼλ��
	std::vector<int> near_cubes_recv_num;
	std::vector<std::vector<int>> far_cubes_process_index;//��¼Զ�������ÿһ�����̵���ʼλ��
	std::vector<std::vector<std::vector<int>> > local_cubes_sent;
	std::vector<NearProxyCube> near_cubes;//������ӣ����ڽ��ڣ�
	std::vector<std::vector<int>> near_cubes_sent;//��Ҫ����j�����̷��͵ı��غ��ӱ��
	std::vector<std::vector<int>> far_cubes_recv_num;
};



#endif // OCTREE_H