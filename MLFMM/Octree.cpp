#include "Octree.h"
#include <bitset>
using namespace std;
using namespace Eigen;

size_t MortonCode1D::GetMortonCode(const JD& point) const
{
	size_t mortoncode = 0;
	JD l = start_point, r = end_point;
	while (r - l > length + 1e-7) {
		JD m = (l + r) / 2;
		if (point < m) {
			mortoncode <<= 1;
			r = m;
		}
		else {
			l = m;
			mortoncode <<= 1;
			mortoncode |= 1;
		}
	}
	return mortoncode;
}

std::vector<size_t> OctreeRWG::GetNearNeighbor(size_t mtc, std::unordered_map<size_t, int>& childs_num)
{
	uint32_t x = 0, y = 0, z = 0;
	// 解码莫顿码到x, y, z坐标
	for (int i = 0; i < mortoncode3d.GetLevelNum(); ++i) {
		x |= ((mtc >> (3 * i)) & 1) << i;
		y |= ((mtc >> (3 * i + 1)) & 1) << i;
		z |= ((mtc >> (3 * i + 2)) & 1) << i;
	}
	std::vector<size_t> near_neighbor;
	for (int dx = -1; dx <= 1; ++dx) {
		if (dx < 0 && x < abs(dx)) continue;
		for (int dy = -1; dy <= 1; ++dy) {
			if (dy < 0 && y < abs(dy)) continue;
			for (int dz = -1; dz <= 1; ++dz) {
				if (dx == 0 && dy == 0 && dz == 0) continue;
				size_t neighbor_mtc = (Part1By2(x + dx) | Part1By2(y + dy) << 1 | Part1By2(z + dz) << 2);
				if (childs_num[neighbor_mtc]) near_neighbor.push_back(neighbor_mtc);
			}
		}
	}
	return near_neighbor;
}

std::vector<size_t> OctreeRWG::GetFarNeighbor(size_t mtc, std::unordered_map<size_t, int>& childs_num)
{
	uint32_t x = 0, y = 0, z = 0;
	// 解码莫顿码到x, y, z坐标
	for (int i = 0; i < mortoncode3d.GetLevelNum(); ++i) {
		x |= ((mtc >> (3 * i)) & 1) << i;
		y |= ((mtc >> (3 * i + 1)) & 1) << i;
		z |= ((mtc >> (3 * i + 2)) & 1) << i;
	}
	std::vector<size_t> far_neighbor;
	for (int dx = -3; dx <= 3; ++dx) {
		if (dx < 0 && x < abs(dx)) continue;
		for (int dy = -3; dy <= 3; ++dy) {
			if (dy < 0 && y < abs(dy)) continue;
			for (int dz = -3; dz <= 3; ++dz) {
				if (dx == 0 && dy == 0 && dz == 0 || abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1) continue;
				size_t neighbor_mtc = (Part1By2(x + dx) | Part1By2(y + dy) << 1 | Part1By2(z + dz) << 2);
				if (childs_num[neighbor_mtc]) far_neighbor.push_back(neighbor_mtc);
			}
		}
	}
	return far_neighbor;
}

VecJD3 MortonCode3D::GetPoint(size_t morton_code, int level_index) const
{
	JD length = mortoncode1d[0].length;
	VecJD3 gap{ 0,0,0 };
	int a = mortoncode1d[0].level_num, b = mortoncode1d[1].level_num, c = mortoncode1d[2].level_num;
	for (int i = level_index; i < a; ++i) {
		int j = i - level_index;
		gap[0] += ((morton_code >> (3 * j) & 1) ? 1 : -1) * (1 << i);
	}
	for (int i = level_index; i < b; ++i) {
		int j = i - level_index;
		gap[1] += ((morton_code >> (3 * j + 1) & 1) ? 1 : -1) * (1 << i);
	}
	for (int i = level_index; i < c; ++i) {
		int j = i - level_index;
		gap[2] += ((morton_code >> (3 * j + 2) & 1) ? 1 : -1) * (1 << i);
	}
	return centre + gap * length / 2;
}

VecJD3 MortonCode3D::GetGap(size_t morton_code, int level_index) const
{
	int a = mortoncode1d[0].level_num, b = mortoncode1d[1].level_num, c = mortoncode1d[2].level_num;
	VecJD3 gap{ 0,0,0 };
	if (level_index < a) gap[0] = ((morton_code & 1) ? -1 : 1) * (1 << level_index);
	if (level_index < b) gap[1] = ((morton_code & 2) ? -1 : 1) * (1 << level_index);
	if (level_index < c) gap[2] = ((morton_code & 4) ? -1 : 1) * (1 << level_index);
	
	return gap * mortoncode1d[0].length / 2;
}

VecJD3 MortonCode3D::GetGap2(size_t mtc1, size_t mtc2, int level_index) const
{
	return (GetPoint(mtc2, level_index) - GetPoint(mtc1, level_index));
}

size_t MortonCode3D::GetMortonCode(const VecJD3& point) const
{
	size_t x = mortoncode1d[0].GetMortonCode(point[0]);
	size_t y = mortoncode1d[1].GetMortonCode(point[1]);
	size_t z = mortoncode1d[2].GetMortonCode(point[2]);
	size_t mortoncode3d = Part1By2(x) | Part1By2(y) << 1 | Part1By2(z) << 2;
	return mortoncode3d;
}


void OctreeRWG::Fillcubes()
{
	const int& level_num = mortoncode3d.GetLevelNum();
	actual_L.resize(level_num + 1, 0);
	//return;
	if (mpipre.GetRank() == 0) cout << mpipre.GetRank() << " 层数 " << level_num << endl;
	std::vector<std::vector<Cube> > full_cubes(level_num);
	std::vector<std::unordered_map<size_t, int> > childs_num(level_num);//莫顿码儿子数量
	std::vector<std::unordered_map<size_t, bool> > have_parent(level_num);
	cout << (old_rwg_ptr->edges).size() << endl;
	//获得完整八叉树
	for (int i = 0; i < (old_rwg_ptr->edges).size(); ++i)
	{
		const VecJD3& point = old_rwg_ptr->GetEdgeCenter(i);
		size_t mortoncode = mortoncode3d.GetMortonCode(point);
		for (int j = 0; j < level_num; j++)
		{
			size_t mtc_now = mortoncode >> (3 * j);
			if (!childs_num[j][mtc_now]) {
				Cube cube;
				cube.mtc = mtc_now;
				full_cubes[j].push_back(cube);
			}
			if (j == 0) ++childs_num[j][mtc_now];
			else if (!have_parent[j - 1][mortoncode >> ((j - 1) * 3)]) {
				++childs_num[j][mtc_now];
				have_parent[j - 1][mortoncode >> ((j - 1) * 3)] = 1;
			}
			VecJD3 center_point = mortoncode3d.GetPoint(mtc_now, j);
			int L = GetL((point - center_point).norm() * 2, GlobalParams.k0);
			actual_L[j] = std::max(actual_L[j], L);
			if (j == 0) {
				int LH = GetLH((point - center_point).norm() * 2, GlobalParams.k0);
				actual_L[level_num] = std::max(actual_L[level_num], LH);
			}
		}
		
	}
	if (mpipre.GetRank() == 0) cout << "完整八叉树构建完成" << endl;
	//排序
	cubes.reserve(level_num);
	for (int i = 0; i < level_num; i++) {
		cubes.push_back(std::vector<Cube>());
		std::sort(full_cubes[i].begin(), full_cubes[i].end(), [](const Cube& a, const Cube& b) {
			return a.mtc < b.mtc;
			}
		);
	}
	/*if (mpipre.GetRank() == 1) {
		for (int i = 0; i < 5; i++) cout << full_cubes[1][i].mtc << " ";
		cout << endl;
		for (int i = 0; i < 20; i++) cout << (full_cubes[0][i].mtc >> 3) << " ";
		cout << endl;

		
		for (int i = 0; i < level_num - 1; i++) {
			cout << "level " << i << " " << full_cubes[i + 1].size() << endl;
			int fa_index = 0;
			for (auto& cube : full_cubes[i]) {
				if((cube.mtc >> 3) != full_cubes[i + 1][fa_index].mtc) ++fa_index;
				auto& fa_cube = full_cubes[i + 1][fa_index];
				if ((cube.mtc >> 3) != fa_cube.mtc) {
					cout << "error " << bitset<15>(fa_cube.mtc) << " " << bitset<15>(cube.mtc) << endl;
				}
			}
		}
		cout << "done" << endl;
	}*/
	if (mpipre.GetRank() == 0)cout << mpipre.GetRank() << " 排序完成 " << full_cubes[0].size() <<endl;

	//获得权重
	vector<vector<int>> BP_weights(mpipre.Get_BP_end());
	for (int i = 0; i < mpipre.Get_BP_end(); ++i) {
		BP_weights[i].resize(full_cubes[i].size(), 0);
		if (i == 0) {
			for (int j = 0; j < full_cubes[i].size(); j++) BP_weights[i][j] = childs_num[i][full_cubes[i][j].mtc];
			continue;
		}
		int start = 0;
		for (int j = 0; j < full_cubes[i].size(); j++) {
			size_t fa_mtc = full_cubes[i][j].mtc;
			while (start < full_cubes[i - 1].size() && (full_cubes[i - 1][start].mtc >> 3) == fa_mtc) {
				BP_weights[i][j] += BP_weights[i - 1][start];
				++start;
			}
		}
	}
	vector<int>& BP_weight = BP_weights[mpipre.Get_BP_end() - 1];
	if (mpipre.GetRank() == 0)cout << mpipre.GetRank() << "权重计算完成" << endl;
	//确定本地八叉树
	for (int i = level_num - 1; i >= 0; --i) {
		if (mpipre.is_PWP(i)||level_num==1) {
			cubes[i] = full_cubes[i];
			continue;
		}
		std::vector<Cube> cubes_level;
		std::vector<CubeWieght> cube_weights;
		cubes_level.reserve(cubes[i + 1].size() * 5);
		cube_weights.reserve(cubes[i + 1].size() * 5);
		for (auto& cube_father : cubes[i + 1]) {
			for (int j = 0; j < full_cubes[i].size(); ++j) {
				if ((full_cubes[i][j].mtc >> 3) != cube_father.mtc) continue;
				cubes_level.push_back(full_cubes[i][j]);
				if (i >= mpipre.Get_BP_end() - 1) {
					CubeWieght cube_weight;
					cube_weight.cube_index = cubes_level.size() - 1;
					cube_weight.weight = i == mpipre.Get_BP_end() - 1 ? BP_weight[j] : childs_num[i][full_cubes[i][j].mtc];
					cube_weights.push_back(cube_weight);
				}
			}
		}
		if (mpipre.GetProcessNum(i + 1) == 1) {
			cubes[i] = cubes_level;
			cubes[i].shrink_to_fit();
			continue;
		}
		Greedy(cubes_level, cube_weights, cubes[i], i);
	}
	if(mpipre.GetRank() == 0)cout << mpipre.GetRank() << "本地八叉树确定完成 " << cubes[0].size()<<endl;
	/*if (mpipre.GetRank() == 0) {
		for (int i = 0; i < 5; i++) cout << cubes[1][i].mtc << " ";
		cout << endl;
		for (int i = 0; i < 20; i++) cout << (cubes[0][i].mtc >> 3) << " ";
		cout << endl;

		for (int i = 0; i < level_num - 1; i++) {
			cout << "level " << i << " " << cubes[i + 1].size() << endl;
			int fa_index = 0;
			for (auto& cube : cubes[i]) {
				while ((cube.mtc >> 3) != cubes[i + 1][fa_index].mtc) ++fa_index;
				auto& fa_cube = cubes[i + 1][fa_index];
				if ((cube.mtc >> 3) != fa_cube.mtc) {
					cout << "error " << bitset<15>(fa_cube.mtc) << " " << bitset<15>(cube.mtc) << endl;
				}
			}
		}
	}*/

	//确定近邻次近邻
	for (int i = 0; i < level_num - 1; i++) {
		if (i == 0) {
			unordered_map<size_t, int> near;
			near_cubes.reserve(cubes[i].size());
			for (int j =0; j < cubes[i].size(); j++) {
				std::vector<size_t> near_neighbor = GetNearNeighbor(cubes[i][j].mtc, childs_num[i]);
				for (auto& mtc_nb : near_neighbor) {
					if (near[mtc_nb] != 0) {
						near_cubes[near[mtc_nb] - 1].local_index.push_back(j);
					}
					else {
						NearProxyCube near_proxy_cube;
						near_proxy_cube.mtc = mtc_nb;
						near_proxy_cube.local_index.reserve(8);
						near_proxy_cube.local_index.push_back(j);
						near_cubes.push_back(near_proxy_cube);
						near[mtc_nb] = near_cubes.size();
					}

				}
			}
			near_cubes.shrink_to_fit();
		}
		vector<ProxyCube> proxy_cubes_level;
		proxy_cubes_level.reserve(cubes[i].size());
		unordered_map<size_t, int> far_map;
		for (int j = 0; j < cubes[i].size(); j++) {
			size_t mtc = cubes[i][j].mtc;
			std::vector<size_t> fa_near_nb = GetNearNeighbor(cubes[i][j].mtc >> 3, childs_num[i + 1]);
			sort(fa_near_nb.begin(), fa_near_nb.end());
			std::vector<size_t> far_neighbor = GetFarNeighbor(cubes[i][j].mtc, childs_num[i]);
			/*for (auto& mtc_nb : far_neighbor) {
				if (mortoncode3d.GetGap2(mtc_nb , cubes[i][j].mtc, i).norm() > mortoncode3d.next_near_length_max[i]) {
					cout << mpipre.GetRank() << " error " << i << " " << j << endl;
				}
			}*/
			for (auto& mtc_nb : far_neighbor) {
				if (!std::binary_search(fa_near_nb.begin(), fa_near_nb.end(), mtc_nb >> 3)) continue;
				if (far_map[mtc_nb] != 0) {
					proxy_cubes_level[far_map[mtc_nb] - 1].local_index.push_back(j);
				}
				else {
					ProxyCube proxy_cube;
					proxy_cube.mtc = mtc_nb;
					proxy_cube.local_index.reserve(50);
					proxy_cube.local_index.push_back(j);
					proxy_cubes_level.push_back(proxy_cube);
					far_map[mtc_nb] = proxy_cubes_level.size();
				}
			}
		}
		proxy_cubes_level.shrink_to_fit();
		proxy_cubes.push_back(proxy_cubes_level);
	}
	if(mpipre.GetRank() == 0)cout << mpipre.GetRank() << "近邻确定完成 " << near_cubes.size() << " " << cubes[0].size() << endl;
}

void OctreeRWG::Greedy(std::vector<Cube>& cubes_level, std::vector<CubeWieght>& cube_weights, std::vector<Cube>& cubes_level_new
	,  int level_index)
{
	std::sort(cube_weights.begin(), cube_weights.end(), [](const CubeWieght& a, const CubeWieght& b) {
		return a.weight == b.weight ? a.cube_index < b.cube_index : a.weight > b.weight;
		}
	);
	int process_index = mpipre.GetProcessIndex(level_index + 1);
	cubes_level_new.reserve(cubes_level.size());
	if (level_index == mpipre.Get_BP_end() - 1) {
		int process_num = mpipre.GetProcessNum(level_index + 1);
		vector<int> sum(process_num, 0);

		for (auto& cube_weight : cube_weights) {
			int min_index = 0, min_weight = 1e9;
			for (int i = 0; i < process_num; ++i) {
				if (sum[i] < min_weight) {
					min_index = i;
					min_weight = sum[i];
				}
			}
			sum[min_index] += cube_weight.weight;
			if (process_index == min_index) {
				cubes_level_new.push_back(cubes_level[cube_weight.cube_index]);
			}
		}
	}
	else {
		int sum1 = 0, sum2 = 0;
		for (auto& cube_weight : cube_weights) {
			if (sum1 <= sum2) sum1 += cube_weight.weight;
			else sum2 += cube_weight.weight;
			if (process_index % 2 == 0 && sum1 <= sum2 || process_index % 2 == 1 && sum1 > sum2) {
				cubes_level_new.push_back(cubes_level[cube_weight.cube_index]);
			}
		}
	}
	std::sort(cubes_level_new.begin(), cubes_level_new.end(), [](const Cube& a, const Cube& b) {
		return a.mtc < b.mtc;
		}
	);
	cubes_level_new.shrink_to_fit();
}

void OctreeRWG::GetNear()
{
	local_cubes_sent.reserve(mortoncode3d.GetLevelNum() - 1);
	far_cubes_recv_num.reserve(mortoncode3d.GetLevelNum() - 1);
	far_cubes_process_index.reserve(mortoncode3d.GetLevelNum() - 1);
	for (int i = 0; i < mortoncode3d.GetLevelNum() - 2; i++) {
		//cout << mpipre.GetRank() << " 开始发送" << i << endl;
		vector<size_t> mtc_send(cubes[i].size());
		for(int j = 0; j < cubes[i].size(); j++) mtc_send[j] = cubes[i][j].mtc;
		//将当前层本地盒子数量发给其他进程
		vector<int> cubes_num_process(mpipre.GetSize());
		int cubes_num = cubes[i].size();
		MPI_Allgather(&cubes_num, 1, MPI_INT, cubes_num_process.data(), 1, MPI_INT, MPI_COMM_WORLD);
		//cout<< mpipre.GetRank() <<" ";
		//for (auto& cubes_num_p : cubes_num_process) cout << cubes_num_p << " " ;
		//cout << endl;
		//将本地盒子的莫顿码发给其他进程
		vector<int> displs(mpipre.GetSize());
		for (int j = 0; j < mpipre.GetSize(); j++) {
			displs[j] = j == 0 ? 0 : displs[j - 1] + cubes_num_process[j - 1];
		}
		int all_cubes_num = displs[mpipre.GetSize() - 1] + cubes_num_process[mpipre.GetSize() - 1];
		vector<size_t> mtc_recv(all_cubes_num);
		MPI_Allgatherv(mtc_send.data(), cubes_num, MPI_UNSIGNED_LONG_LONG, 
			mtc_recv.data(), cubes_num_process.data(), displs.data(), MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		//if (i == 2 && mpipre.GetRank() == 0) {
		//	/*for (auto& process : displs) cout << process << " ";
		//	cout << endl;*/
		//	/*for (int j = 0;j < mtc_recv.size();j++) {
		//		if (mtc_recv[j] == 15) {
		//			cout << "有！！！！" << " " << j << endl;
		//			
		//		}
		//	}*/
		//	/*for (int j = 0; j < mtc_recv.size(); j++) {
		//		cout << j << "        " << mtc_recv[j] << endl;
		//	}*/
		//}
		//近邻盒子
		if (i == 0) {
			for (auto& near_nb : near_cubes) {
				for (int j = 0; j < mpipre.GetSize(); j++) {
					auto st = mtc_recv.begin() + displs[j];
					auto ed = st + cubes_num_process[j];
					int m0 = lower_bound(st, ed, near_nb.mtc) - mtc_recv.begin();
					if (m0 < all_cubes_num && mtc_recv[m0] == near_nb.mtc) {
						near_nb.process_index = j;
						break;
					}
				}
			}

			std::sort(near_cubes.begin(), near_cubes.end(), [](const NearProxyCube& a, const NearProxyCube& b) {
				return a.process_index == b.process_index ? a.mtc < b.mtc : a.process_index < b.process_index;
				}
			);
			

			near_cubes_process_index.resize(mpipre.GetSize() + 1, 0);
			for (int j = 0; j < mpipre.GetSize(); ++j) {
				near_cubes_process_index[j + 1] = near_cubes_process_index[j];
				while(near_cubes_process_index[j + 1] < near_cubes.size() && near_cubes[near_cubes_process_index[j + 1]].process_index == j) ++near_cubes_process_index[j + 1];
			}
			//cout << mpipre.GetRank() << "每一进程代理盒子起始位置：" << near_cubes_process_index[0] << " " << near_cubes_process_index[1] << endl;
			near_cubes_sent.resize(mpipre.GetSize());
			int index = 0;
			for (int j = 0; j < mpipre.GetSize(); j++) {
				set<int> cube_be_sent;
				//cout << mpipre.GetRank() << "  " << j << " " << near_proxy_cubes[0].process_index << endl;
				while (index < near_cubes.size() && near_cubes[index].process_index == j) {
					for (auto& local_index : near_cubes[index].local_index) cube_be_sent.insert(local_index);
					index++;
					//if (mpipre.GetRank() == 0) cout << index << " " << near_proxy_cubes[index].process_index << " " << near_proxy_cubes.size() << endl;
				}
				near_cubes_sent[j] = std::vector<int>(cube_be_sent.begin(), cube_be_sent.end());
				//cout << mpipre.GetRank() << "发送的盒子数量：" << j << " " << near_cubes_sent[j].size() << endl;
			}
		}
		//cout<< mpipre.GetRank() << " ??? " << endl;
		//确定每一个代理盒子所属进程
		auto& transfer_process = mpipre.GetTransferProcess(i);
		std::vector<int> far_cube_recv_num(mpipre.GetSize(), 0);
		//cout << mpipre.GetRank() << " ??? " << transfer_process.size() << " " << proxy_cubes.size() << endl;
		for (auto& proxy_cube : proxy_cubes[i]) {
			for (int j = 0;j < transfer_process.size();j++) {
				int process_index = transfer_process[j];
				auto st = mtc_recv.begin() + displs[process_index];
				auto ed = st + cubes_num_process[process_index];
				int m0 = lower_bound(st, ed, proxy_cube.mtc) - mtc_recv.begin();
				if (m0 < all_cubes_num && mtc_recv[m0] == proxy_cube.mtc) {
					//proxy_cube.my_index = m0 - displs[j];
					proxy_cube.process_index = process_index;
					++far_cube_recv_num[process_index];
					break;
				}
			}
		}
		far_cubes_recv_num.push_back(far_cube_recv_num);
		//cout << mpipre.GetRank() << " ???? " << endl;
		//对代理盒子排序
		std::sort(proxy_cubes[i].begin(), proxy_cubes[i].end(), [](const ProxyCube& a, const ProxyCube& b) {
			return a.process_index == b.process_index ? a.mtc < b.mtc : a.process_index < b.process_index;
			}
		);
		vector<int> far_cubes_process(mpipre.GetSize() + 1, 0);
		for (int j = 0; j < mpipre.GetSize(); ++j) {
			far_cubes_process[j + 1] = far_cubes_process[j] + far_cube_recv_num[j];
			/*far_cubes_process[j + 1] = far_cubes_process[j];
			while (far_cubes_process[j + 1] < proxy_cubes[i].size() && proxy_cubes[i][far_cubes_process[j + 1]].process_index == j) ++far_cubes_process[j + 1];*/
		}
		far_cubes_process_index.push_back(far_cubes_process);
		//cout << mpipre.GetRank() << " ????? " << endl;
		//确定需要发送到第j个进程的本地盒子序号
		std::vector<std::vector<int> > cubes_sent_level(transfer_process.size());
		int index = 0;
		for (int j = 0; j < transfer_process.size(); j++) {
			set<int> cube_be_sent;
			while (index < proxy_cubes[i].size() && proxy_cubes[i][index].process_index == transfer_process[j]) {
				for (auto& local_index : proxy_cubes[i][index].local_index) cube_be_sent.insert(local_index);
				index++;
			}
			cubes_sent_level[j] = std::vector<int>(cube_be_sent.begin(), cube_be_sent.end());
			//if (i == 2) {
			//	/*if (mpipre.GetRank() == 0) {
			//		for (auto& proxy_cube : proxy_cubes[i]) {
			//			cout << proxy_cube.process_index << " " << proxy_cube.mtc << endl;
			//			for (auto& local_index : proxy_cube.local_index) cout << local_index << " ";
			//			cout << "done!" << endl;
			//		}
			//	}*/
			//	
			//	cout << mpipre.GetRank() << " 发送的盒子数量：" << j << " " << cubes_sent_level[j].size() << " " << transfer_process[j] << endl;
			//}
		}
		local_cubes_sent.push_back(cubes_sent_level);
	}
	if(mpipre.GetRank() == 0)cout<< mpipre.GetRank() << " 获得近远邻完成 " << endl;
}

void OctreeRWG::change_rwgs()
{
	auto& old_rwgs = *old_rwg_ptr;
	int edges_num = old_rwgs.edges.size();
	cube_rwgs_num.resize(cubes[0].size(), 0);
	near_cube_rwgs_num.resize(near_cubes.size(), 0);
	int sum1 = 0, sum2 = 0;
	vector<vector<int>> cube_rwgs_index(cubes[0].size()+ near_cubes.size());
	int sum3 = 0;
	//cout << mpipre.GetRank() << " ? " << edges_num << endl;
	for (int i = 0; i < edges_num; i++) {
		const VecJD3& point = old_rwg_ptr->GetEdgeCenter(i);
		size_t mtc = mortoncode3d.GetMortonCode(point);
		int m0 = lower_bound(cubes[0].begin(), cubes[0].end(), mtc,[](const Cube& a, size_t b){
			return a.mtc < b; }) - cubes[0].begin();
		if (m0 < cubes[0].size() && mtc == cubes[0][m0].mtc) {
			cube_rwgs_index[m0].push_back(i);
			++cube_rwgs_num[m0];
			++sum1;
		}
		for (int j = 0; j < mpipre.GetSize(); ++j) {
			auto st = near_cubes.begin() + near_cubes_process_index[j];
			auto ed = near_cubes.begin() + near_cubes_process_index[j + 1];
			int m1 = lower_bound(st, ed, mtc, [](const NearProxyCube& a, size_t b) {return a.mtc < b; }) - near_cubes.begin();
			if (m1 < near_cubes.size() && mtc == near_cubes[m1].mtc) {
				if (near_cubes[m1].process_index != mpipre.GetRank()) {
					cube_rwgs_index[cubes[0].size() + m1].push_back(i);
					++sum2;
				}
				++near_cube_rwgs_num[m1];
				break;
			}
		}
		//int m1 = lower_bound(near_cubes.begin(), near_cubes.end(), mtc,[](const NearProxyCube& a, size_t b){return a.mtc < b; }) - near_cubes.begin();
		//if ( m1 < near_cubes.size() && mtc == near_cubes[m1].mtc) {
		//	if (near_cubes[m1].process_index != mpipre.GetRank()) {
		//		cube_rwgs_index[cubes[0].size() + m1].push_back(i);
		//		++sum2;
		//	}
		//	
		//	++near_cube_rwgs_num[m1];
		//	//++sum3;
		//}
	}
	cube_rwgs_dif.resize(cubes[0].size() + 1);
	for (int i = 0; i <= cubes[0].size(); i++) cube_rwgs_dif[i] = i == 0 ? 0 : cube_rwgs_dif[i - 1] + cube_rwgs_num[i - 1];

	near_cubes_recv_num.resize(mpipre.GetSize(), 0);
	int near_cubes_index = 0;
	for (int j = 0; j < mpipre.GetSize(); j++) {
		while (near_cubes_index < near_cubes.size() && near_cubes[near_cubes_index].process_index == j) {
			near_cubes_recv_num[j] += near_cube_rwgs_num[near_cubes_index++];
		}
		//cout << "本地进程" << mpipre.GetRank() << " 从进程 " << j << " 接收 " << near_cubes_recv_num[j] << endl;
	}
	
	//cout << mpipre.GetRank() << " rwg分配给本地盒子 " << sum1 << " " << sum2 << " "<< local_rwgs_num() <<endl;
	rwg.edges.reserve(sum1 + sum2);
	rwg.points.reserve((sum1 + sum2) / 3);
	rwg.triangles.reserve((sum1 + sum2) / 1.5);
	unordered_map<int, int> point_map{};
	unordered_map<int, int> triangle_map{};

	for (auto& cube_index : cube_rwgs_index) {
		for (auto& rwg_index : cube_index) {
			rwg.edges.push_back(old_rwgs.edges[rwg_index]);
			Vector2i& edge = rwg.edges.back();
			for (int i = 0; i < 2; i++) {
				if (triangle_map[edge[i]] == 0) {
					rwg.triangles.push_back(old_rwgs.triangles[edge[i]]);
					triangle_map[edge[i]] = rwg.triangles.size();
					edge[i] = rwg.triangles.size() - 1;
					for (int j = 0; j < 3; j++) {
						int& point_index = rwg.triangles[edge[i]][j];
						if (point_map[point_index] == 0) {
							rwg.points.push_back(old_rwgs.points[point_index]);
							point_map[point_index] = rwg.points.size();
						}
						point_index = point_map[point_index] - 1;
					}
				}
				else edge[i] = triangle_map[edge[i]] - 1;

			}
		}
	}

	delete old_rwg_ptr;
	rwg.edges.shrink_to_fit();
	rwg.points.shrink_to_fit();
	rwg.triangles.shrink_to_fit();
	if (mpipre.GetRank() == 0) cout<<mpipre.GetRank() << "rwg重分配完成 " << rwg.edges.size()<<" "<< local_rwgs_num()<<endl;
	//cout << mpipre.GetRank() << "rwg重分配完成" << endl;
	//cout << "rwg重分配完成" << mpipre.GetRank() << " " << rwgs.size() << endl;
	//cubes_sort();
}

void OctreeRWG::Fillb(JD theta, JD phi)
{
	//cout<< mpipre.GetRank() << " Fillb开始 " << endl;
	rwg.set_wave(theta, phi);
	int size = local_rwgs_num();
	b.setZero(size);
	for (int i = 0; i < size; i++) {
		b[i] = rwg.Getbi(i);
	}

	//JD ct = cos(theta * rad), st = sin(theta * rad), cp = cos(phi * rad), sp = sin(phi * rad);
	//Vector3d r = Vector3d(st * cp, st * sp, ct);//方向相反
	//int size = local_rwgs_num();
	//b.setZero(size);
	//MatrixXd T(2, 3);
	//T << ct * cp, ct* sp, -st,
	//	-sp, cp, 0;
	////T.transposeInPlace();
	//Vector2d polor{ -1.0,0.0 };
	//Vector3d plane_wave = T.transpose() * polor;
	//for (int i = 0; i < size; i++) {
	//	Eigen::Matrix<JD, 3, GLPN2>& Ji = rwg.J_GL[i];
	//	for (int m = 0; m < 2; ++m) {
	//		JD pmm = m ? -1.0 : 1.0;
	//		for (int p = 0; p < GLPN; ++p) {
	//			Vector3d Jm = pmm * (Ji.col(m * GLPN + p) - rwg.points[rwg.vertex_edges[i][m]]);
	//			JD rpr = GlobalParams.k0 * Ji.col(m * GLPN + p).dot(r);
	//			CP ejkr = CP(cos(rpr), sin(rpr));
	//			b[i] += GL_points[p][0] * Jm.dot(plane_wave) * ejkr;
	//		}
	//	}
	//	b[i] *= rwg.edges_length[i] * 0.5;
	//}
	//cout<< mpipre.GetRank() << " Fillb完成 " << endl;
}