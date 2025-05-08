#include "MatrixPre.h"

using namespace std;
using namespace Eigen;

void MatrixPre::GetNearNb()
{
	
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	auto& near_cubes = octree.near_cubes;
	JD norm_sum = 0;
	//cout<<mpipre.GetRank()<<"开始填充近作用矩阵 "<< cube_floor.size() <<endl;
	Zinv_list.reserve(cube_floor.size());
	for (int i = 0; i < cube_floor.size(); i++) {
		int st = octree.cube_rwgs_dif[i], ed = st + octree.cube_rwgs_num[i], size = octree.cube_rwgs_num[i];
		//cout << i << " " << st << " " << ed << endl;
		MatCP Zs(size, size);
		for (int k = 0; k < size; k++) {
			for (int j = k; j < size; j++) {
				Zs(j, k) = octree.rwg.GetZij(j + st, k + st);
				Zs(k, j) = Zs(j, k);
			}
		}
		Eigen::PartialPivLU<MatCP> Zinv(Zs);
		//cout <<"矩阵范数"<< Zs.squaredNorm() << endl;
		norm_sum += size * size;
		Zinv_list.push_back(Zinv);
	}

	int matrix_size = 0;
	for (auto& near_cube : near_cubes) matrix_size += near_cube.local_index.size();
	near_matrix_list.reserve(matrix_size);


	int proxy_rwgs_st = octree.cube_rwgs_dif[cube_floor.size() - 1] + octree.cube_rwgs_num[cube_floor.size() - 1];//代理rwg开始位置
	//填充每一个近盒子和近邻的关系矩阵
	for (int k = 0; k < near_cubes.size(); k++) {
		int rwgs_st = proxy_rwgs_st, rwgs_end = proxy_rwgs_st + octree.near_cube_rwgs_num[k], num = octree.near_cube_rwgs_num[k];
		if (near_cubes[k].process_index == mpipre.GetRank()) {
			int m0 = lower_bound(cube_floor.begin(), cube_floor.end(), near_cubes[k].mtc, [](const Cube& cube, const size_t& mtc) {return cube.mtc < mtc; }) - cube_floor.begin();
			rwgs_st = octree.cube_rwgs_dif[m0];
			rwgs_end = octree.cube_rwgs_dif[m0] + octree.cube_rwgs_num[m0];
			num = octree.cube_rwgs_num[m0];
		}
		//cout << k << "近邻 " << rwgs_st << " " << rwgs_end << " " << near_cubes[k].local_index.size() << " " << octree.near_cube_rwgs_num[k] << endl;
		for (auto& index : near_cubes[k].local_index) {
			int local_rwgs_end = octree.cube_rwgs_dif[index] + octree.cube_rwgs_num[index];
			/*if (k == 0) {
				cout << index << "近邻 " << cube_floor[index].mtc << " " << octree.cube_rwgs_dif[index] << " " << local_rwgs_end << " "<< near_cubes[k].mtc<<endl;
				cout << octree.mortoncode3d.GetGap2(cube_floor[index].mtc, near_cubes[k].mtc, 0).norm() / (0.251 * GlobalParams.lam) << endl;
			}*/
			MatCP Zn(octree.cube_rwgs_num[index], num);
			for (int i = octree.cube_rwgs_dif[index]; i < local_rwgs_end; i++) {
				for (int j = rwgs_st; j < rwgs_end; j++) {
					Zn(i - octree.cube_rwgs_dif[index], j - rwgs_st) = octree.rwg.GetZij(i, j);
				}
			}
			near_matrix_list.push_back(Zn);
			norm_sum += Zn.size();
			near_cubes[k].index.push_back(near_matrix_list.size() - 1);
		}
		proxy_rwgs_st += near_cubes[k].process_index == mpipre.GetRank() ? 0 : octree.near_cube_rwgs_num[k];
	}
	/*JD all_sum = 0;
	MPI_Reduce(&norm_sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(mpipre.GetRank() == 0) cout <<"总范数"<< norm_sum << endl;*/
	JD mem_all = mpiout(norm_sum, mpipre);
	if (mpipre.GetRank() == 0)cout << mpipre.GetRank() << "近作用矩阵填充完成 " << near_matrix_list.size() << " " << MemUsed(mem_all) * 2 << "GB" << endl;
}

void MatrixPre::NearProd(const VecCP& x, VecCP& b)
{
	//cout << mpipre.GetRank() << " ?? " << mpi_mem_end_ptr- mpi_mem_st_ptr<<endl;
	int rank = mpipre.GetRank();
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	auto& near_cubes = octree.near_cubes;
	queue<MPI_Request> send_requests;
	queue<MPI_Request> recv_requests;
	auto& cubes_sent_process = octree.near_cubes_sent;
	CP* st = mpi_mem_st_ptr;
	for (int i = 0; i < mpipre.GetSize(); i++) {
		int m0 = (rank + i + 1) % mpipre.GetSize();
		auto& cubes_sent = cubes_sent_process[m0];
		//cout << mpipre.GetRank() << " 向 " << m0 << " 发送 " << cubes_sent_process[m0].size() << endl;
		if (cubes_sent.empty()) continue;
		int shift = 0;
		for (auto& j : cubes_sent) {
			Map<VecCP> x_sent(st + shift, octree.cube_rwgs_num[j]);
			x_sent = x.segment(octree.cube_rwgs_dif[j], octree.cube_rwgs_num[j]);
			shift += octree.cube_rwgs_num[j];
			/*if (x_sent.squaredNorm() > 1e9) {
				cout <<mpipre.GetRank()<<" "<< m0<<" "<< j << " ??? " << octree.cube_rwgs_num[j] << endl;
			}*/
		}
		send_requests.push(MPI_Request());
		Map<VecCP> hh2(st, shift);
		//cout << mpipre.GetRank() << " 发送范数 " << hh2.squaredNorm() << " " << hh2.topRows(2).transpose() << " " << shift << " "<< hh2.bottomRows(2).transpose()<<endl;
		MPI_Isend(st, shift , MPI_DOUBLE_COMPLEX, m0, 0, MPI_COMM_WORLD, &send_requests.back());
		st += shift;
	}
	//cout << mpipre.GetRank() << " ??? " << octree.near_cubes_recv_num.size()<< endl;
	CP* st_temp = st;
	for (int i = 0; i < mpipre.GetSize(); i++) {
		int m0 = (rank - i - 1 + mpipre.GetSize()) % mpipre.GetSize();
		//cout << mpipre.GetRank() << " 从 " << m0 << " 接收  " << octree.near_cubes_recv_num[m0] << " " << mpi_mem_end_ptr - st << endl;
		if(octree.near_cubes_recv_num[m0] == 0) continue;
		recv_requests.push(MPI_Request());
		MPI_Irecv(st, octree.near_cubes_recv_num[m0], MPI_DOUBLE_COMPLEX, m0, 0, MPI_COMM_WORLD, &recv_requests.back());
		st += octree.near_cubes_recv_num[m0];
	}
	//cout << "通信数量" << mpipre.GetSize() << " " << send_requests.size() << " " << recv_requests.size() << endl;
	for (int i = 0; i < mpipre.GetSize(); i++) {
		int m0 = (rank - i - 1 + mpipre.GetSize()) % mpipre.GetSize();
		if (octree.near_cubes_recv_num[m0] == 0) continue;
		MPI_Wait(&recv_requests.front(), MPI_STATUS_IGNORE);
		Map<VecCP> hh2(st_temp, octree.near_cubes_recv_num[m0]);
		//cout << mpipre.GetRank() << " 接收范数" << hh2.squaredNorm() << " "<< hh2.topRows(3).transpose() << endl;
		for (int k = octree.near_cubes_process_index[m0]; k < near_cubes.size() && near_cubes[k].process_index == m0; ++k) {//第near_process_st个盒子接收到谱后依次关系到对应的本地盒子
			Map<VecCP> x_recv(st_temp, octree.near_cube_rwgs_num[k]);//第k个近邻盒子的收到的谱
			//cout << "收到" << near_cubes[k].index.size() << " " << octree.near_cube_rwgs_num[k] << " " << k << endl;
			for (int j = 0; j < near_cubes[k].index.size(); j++) {
				int cube_index = near_cubes[k].local_index[j];//对应本地盒子编号
				int rwgs_st = octree.cube_rwgs_dif[cube_index];
				b.segment(rwgs_st, octree.cube_rwgs_num[cube_index]).noalias() += near_matrix_list[near_cubes[k].index[j]] * x_recv;
				//cout << j << endl;
			}
			st_temp += octree.near_cube_rwgs_num[k];
		}
		recv_requests.pop();
	}
	//cout << "done!" << endl;
}

void MatrixPre::SelfProd(const VecCP& x, CP* b_str)
{
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	for (int i = 0; i < cube_floor.size(); i++) {
		int st = octree.cube_rwgs_dif[i];
		Map<VecCP> b(b_str + st, octree.cube_rwgs_num[i]);
		b.noalias() = Zinv_list[i].solve(x.segment(st, octree.cube_rwgs_num[i]));
	}
}

void MatrixPre::GetDecomposes()
{
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	int level_num = spectrum_pre.level_num;
	int spectrum_num = spectrum_pre.k_vecs[level_num].size();
	//cout << "聚合" << spectrum_num << endl;
	decomposes.reserve(cube_floor.size());
	const RWG& rwg = octree.rwg;
	int edge_index = 0;
	for (int i = 0; i < cube_floor.size(); i++) {
		auto& cube = cube_floor[i];
		VecJD3 point = octree.mortoncode3d.GetPoint(cube.mtc, 0);
		MatCP decompose(octree.cube_rwgs_num[i], spectrum_num * 2);
		//cout << i << " " << octree.cube_rwgs_num[i] << endl;
		for (int j = 0; j < octree.cube_rwgs_num[i]; j++) {
			const Eigen::Matrix<JD, 3, GLPN2>& Jj = rwg.J_GL[edge_index];
			for (int row = 0; row < spectrum_num; row++) {
				VecCP3 e1 = VecCP3::Zero();
				VecCP3 e2 = VecCP3::Zero();
				for (int m = 0; m < 2; ++m)
				{
					JD pmm = m ? -1.0 : 1.0;
					for (int p = 0; p < GLPN; p++) {
						Vector3d Jm = pmm * (Jj.col(m * GLPN + p) - rwg.points[rwg.vertex_edges[edge_index][m]]);
						JD inrpr = -(spectrum_pre.k_vecs[level_num][row]).dot(Jj.col(m * GLPN + p) - point);
						JD ct = cos(inrpr), st = sin(inrpr);
						Vector3d n_unit = rwg.tri_normal[rwg.edges[edge_index][m]];
						VecCP3 ej = GL_points[p][0] * rwg.edges_length[edge_index] * (Jm)*CP(ct, st);
						VecCP3 eh = GL_points[p][0] * rwg.edges_length[edge_index] * (Jm.cross(n_unit)) * CP(ct, st);
						e1 += ej;// +CPD(0.0, k_unit[0].norm() / (4.0 * pi)) * mj;
						e2 += eh;
					}

				}
				e2 = (spectrum_pre.k_vecs[level_num][row]).cross(e2).conjugate();
				decompose.block(j, row, 1, 1) = spectrum_pre.TCtoS[row].row(0) * 0.5 * (rwg.alpha * e1 - (1.0 - rwg.alpha)/ GlobalParams.k0 * e2);
				decompose.block(j, row + spectrum_num, 1, 1) = spectrum_pre.TCtoS[row].row(1) * 0.5 * (rwg.alpha * e1 - (1.0 - rwg.alpha)/ GlobalParams.k0 * e2);
				/*decompose.block(row, j, 1, 1) = spectrum_pre.TCtoS[row].row(0) * 0.5 * e1;
				decompose.block(row + spectrum_num, j, 1, 1) = spectrum_pre.TCtoS[row].row(1) * 0.5 * e1;*/
			}
			++edge_index;
		}
		decompose *= CP(0.0, Zf * GlobalParams.k0 / (4.0 * pi));
		decomposes.push_back(decompose);
	}
}

void MatrixPre::GetAggregations()
{
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	int level_num = spectrum_pre.level_num;
	int spectrum_num = spectrum_pre.k_vecs[level_num].size();
	JD mem_sum = 0;
	aggregations.reserve(cube_floor.size());
	const RWG& rwg = octree.rwg;
	int edge_index = 0;
	for (int i = 0; i < cube_floor.size(); i++) {
		auto& cube = cube_floor[i];
		VecJD3 point = octree.mortoncode3d.GetPoint(cube.mtc, 0);
		MatCP aggregation(spectrum_num * 2, octree.cube_rwgs_num[i]);
		//cout << i << " " << octree.cube_rwgs_num[i] << endl;
		for (int j = 0; j < octree.cube_rwgs_num[i]; j++) {
			const Eigen::Matrix<JD, 3, GLPN2>& Jj = rwg.J_GL[edge_index];
			for (int row = 0; row < spectrum_num; row++) {
				VecCP3 e1 = VecCP3::Zero();
				VecCP3 e2 = VecCP3::Zero();
				for (int m = 0; m < 2; ++m)
				{
					JD pmm = m ? -1.0 : 1.0;
					for (int p = 0; p < GLPN; p++) {
						Vector3d Jm = pmm * (Jj.col(m * GLPN + p) - rwg.points[rwg.vertex_edges[edge_index][m]]);
						JD inrpr = (spectrum_pre.k_vecs[level_num][row]).dot(Jj.col(m * GLPN + p) - point);
						JD ct = cos(inrpr), st = sin(inrpr);
						//Vector3d n_unit = rwg.tri_normal[rwg.edges[edge_index][m]];
						//Vector3d mc = k_unit[row].normalized().cross(rwgs[rwgs_index].vec_c[k]);
						//Vector3d mc = 1.0 / Zf * spectrum_pre.k_vecs[level_num][row].normalized().cross(n_unit.cross(Jm));
						//Vector3cd mj = points[k % GLPN][0] * rwgs[rwgs_index].length * mc * CPD(ct, st);
						VecCP3 ej = GL_points[p][0] * rwg.edges_length[edge_index] * (Jm) * CP(ct, st);
						//VecCP3 eh = GL_points[p][0] * rwg.edges_length[edge_index] * (Jm.cross(n_unit)) * CP(ct, st);
						e1 += ej;// +CPD(0.0, k_unit[0].norm() / (4.0 * pi)) * mj;
						//e2 += eh;
					}

				}
				//e2 = (spectrum_pre.k_vecs[level_num][row]).cross(e2);
				aggregation.block(row, j, 1, 1) = spectrum_pre.TCtoS[row].row(0) * 0.5 * e1;
				aggregation.block(row + spectrum_num, j, 1, 1) = spectrum_pre.TCtoS[row].row(1) * 0.5 * e1;
			}
			++edge_index;
		}
		mem_sum += aggregation.size();
		aggregations.push_back(aggregation);
	}
	TWIP twip(spectrum_pre.thetas[0], spectrum_pre.thetas[level_num], spectrum_pre.phis_num[0], spectrum_pre.phis_num[level_num]);
	twip.GetMatrix(0, spectrum_pre.thetas_num[0], 0, spectrum_pre.thetas_num[level_num]);
	F = twip.First;
	S = twip.Second;
	aF = twip.First.transpose();
	aS = twip.Second.transpose();
	JD mem_all = mpiout(mem_sum, mpipre);
	if (MPIpre().GetRank() == 0)cout << "Aggregation Done! " << spectrum_pre.thetas_num[level_num] << " " << MemUsed(mem_all) * 4 << "GB" << endl;
}

void MatrixPre::AggregationProd(CP* J_ptr, bool judge)
{
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	int sptm_num0 = spectrum_pre.GetSptmNumLevel(0);
	int spectrum_num = spectrum_pre.k_vecs[spectrum_pre.level_num].size();
	//cout << "波谱数量" << sptm_num0 << endl;
	int offset = 0;
	double sum = 0;
	for (auto i = 0; i < cube_floor.size(); ++i)
	{
		auto& aggregation = aggregations[i];
		//cout << i << " " << offset << endl;
		Map<VecCP> spectrum_theta(cube_floor[i].sptmtheta, sptm_num0);
		Map<VecCP> spectrum_phi(cube_floor[i].sptmtheta + sptm_num0, sptm_num0);
		Map<VecCP> x(J_ptr + offset, octree.cube_rwgs_num[i]);
		//cout <<i<<" "<< offset << endl;
		if (judge) {
			/*if (i == 0) {
				hhc(0, cube_floor[i].sptmtheta);
			}*/
			x.noalias() += (decomposes[i].leftCols(spectrum_num) * (aF * (aS * spectrum_theta))
				+ decomposes[i].rightCols(spectrum_num) * (aF * (aS * spectrum_phi)));
			//if (i == 0) cout <<"配置后范数："<< x.squaredNorm() << endl;
		}
		else {
			spectrum_theta.noalias() = S * (F * (aggregation.topRows(spectrum_num) * x));
			spectrum_phi.noalias() = S * (F * (aggregation.bottomRows(spectrum_num) * x));
		}
		offset += octree.cube_rwgs_num[i];
	}
	
	
	/*if (judge) {
		Map<VecCP> hh(J_ptr, GetRwgNum());
		cout << "配置后范数：" << hh.squaredNorm() << endl;
	}
	else {
		Map<VecCP> spectrum_theta(cube_floor[0].sptmtheta, sptm_num0 * 2 * cube_floor.size());
		cout<<" 聚合后范数："<< spectrum_theta.squaredNorm() << endl;
	}*/
}

void MatrixPre::GetInterpolations()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	Fu_list.reserve(level_num);
	Fm_list.reserve(level_num);
	Fd_list.reserve(level_num);
	aF_list.reserve(level_num);
	aSu_list.reserve(level_num);
	aSm_list.reserve(level_num);
	S_list.reserve(level_num);
	aSd_list.reserve(level_num);
	for (int i = 0; i < level_num - 1; i++) {
		int fa_theta_st = spectrum_pre.ip_thetas_st[i], fa_theta_end = spectrum_pre.ip_thetas_end[i];
		int son_theta_st = spectrum_pre.sptm_thetas_st[i], son_theta_end = spectrum_pre.sptm_thetas_end[i];
		int fa_phi_num = spectrum_pre.phis_num[i + 1], son_phi_num = spectrum_pre.phis_num[i];

		int son_st_shift = son_theta_st ? TWIP::p * son_phi_num : 0;
		int son_end_shift = son_theta_end < spectrum_pre.thetas_num[i] ? TWIP::p * son_phi_num : 0;
		int fa_st_shift = son_theta_st ? TWIP::pp * fa_phi_num : 0;
		int fa_end_shift = son_theta_end < spectrum_pre.thetas_num[i] ? TWIP::pp * fa_phi_num : 0;
		int temp_shift = son_theta_st ? TWIP::pp * son_phi_num : 0;
		int actual_fa_spectrum_num = (fa_theta_end - fa_theta_st) * fa_phi_num;
		int actual_temp_spectrum_num = (fa_theta_end - fa_theta_st) * son_phi_num;
		int actual_son_spectrum_num = (son_theta_end - son_theta_st) * son_phi_num;

		TWIP twip(spectrum_pre.thetas[i + 1], spectrum_pre.thetas[i], spectrum_pre.phis_num[i + 1], spectrum_pre.phis_num[i]);
		twip.GetMatrix(fa_theta_st, fa_theta_end, son_theta_st, son_theta_end);

		Fu_list.push_back(twip.First.middleRows(temp_shift, actual_temp_spectrum_num).leftCols(son_st_shift));
		Fm_list.push_back(twip.First.middleRows(temp_shift, actual_temp_spectrum_num).middleCols(son_st_shift, actual_son_spectrum_num));
		Fd_list.push_back(twip.First.middleRows(temp_shift, actual_temp_spectrum_num).rightCols(son_end_shift));
		S_list.push_back(twip.Second.block(fa_st_shift, temp_shift, actual_fa_spectrum_num, actual_temp_spectrum_num));
		aSu_list.push_back(twip.Second.transpose().leftCols(fa_st_shift));
		aSm_list.push_back(twip.Second.transpose().middleCols(fa_st_shift, actual_fa_spectrum_num));
		aSd_list.push_back(twip.Second.transpose().rightCols(fa_end_shift));
		aF_list.push_back(twip.First.transpose().middleRows(son_st_shift, actual_son_spectrum_num));
	}
	if (mpipre.GetRank() == 0) cout << "获得插值矩阵" << endl;
}

void MatrixPre::GetPhaseShifts()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	phase_shifts.reserve(level_num);

	for (int i = 0; i < level_num - 1; ++i)
	{
		//unordered_map<int, int> phase_shift_map_now;
		vector<VectorXcd> phase_shift_now;
		phase_shift_now.resize(8);

		int spectrum_father_num = (spectrum_pre.ip_thetas_end[i] - spectrum_pre.ip_thetas_st[i]) * spectrum_pre.phis_num[i + 1];
		auto& cubes_fa = octree.GetCubesLevel(i + 1);
		auto& cubes_son = octree.GetCubesLevel(i) ;
		int son_index = 0;
		for (int j = 0; j < cubes_fa.size(); ++j)
		{
			while (son_index < cubes_son.size() && (cubes_son[son_index].mtc >> 3) == cubes_fa[j].mtc)
			{
				VecJD3 r_phase = octree.mortoncode3d.GetGap(cubes_son[son_index].mtc, i);
				int phase_shift_index = (cubes_son[son_index].mtc & 7);
				++son_index;
				/*int hh = son_index - 1;
				if (j == 0) {
					cout<< hh <<"gg "<<((cubes_son[hh].mtc >> 3) == cubes_fa[j].mtc)<<" "<< cubes_son[hh].mtc<<" "<< cubes_fa[j].mtc<<" "<< phase_shift_index<<" "<<
						phase_shift_now[phase_shift_index].size()<<endl;
				}*/
				if (phase_shift_now[phase_shift_index].size() > 0) continue;
				/*cout << r_phase.transpose() / spectrum_pre.length << endl << octree.mortoncode3d.GetPoint(cubes_fa[j].mtc, i + 1).transpose() / spectrum_pre.length << endl <<
					octree.mortoncode3d.GetPoint(cubes_son[hh].mtc, i).transpose() / spectrum_pre.length << endl
					<< (octree.mortoncode3d.GetPoint(cubes_fa[j].mtc, i + 1) - octree.mortoncode3d.GetPoint(cubes_son[hh].mtc, i)).transpose() / spectrum_pre.length << " " << phase_shift_index << endl <<
					cubes_son[hh].mtc << " " << cubes_fa[j].mtc << " " << (cubes_son[hh].mtc >> 3) << endl;*/
				phase_shift_now[phase_shift_index].resize(spectrum_father_num);
				for (int l = 0; l < spectrum_father_num; ++l) {
					JD kdotr = -(spectrum_pre.k_vecs[i + 1][l]).dot(r_phase);
					JD ct = cos(kdotr), st = sin(kdotr);
					phase_shift_now[phase_shift_index][l] = CP(ct, st);
				}
			}
		}
		phase_shifts.push_back(phase_shift_now);
	}
	if (mpipre.GetRank() == 0) cout << "获得相位转移矩阵" << endl;
}

void MatrixPre::GetTransfers()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	transfers.reserve(level_num);
	int mem_sum = 0;

	for (int i = 0; i < level_num; ++i)
	{
		auto& proxy_cubes = octree.GetProxyCubesLevel(i);
		auto& cubes = octree.GetCubesLevel(i);
		std::vector<VecCP> transfers_level;
		std::vector<VecJD3> transfers_D;
		transfers_level.reserve(316);
		//int L = GetL(spectrum_pre.length * sqrt(3.0) * (1 << i), GlobalParams.k0);
		int L = octree.actual_L[i];
		const int itp_local_st = i == 0 ? 0 : (spectrum_pre.sptm_thetas_st[i] - spectrum_pre.ip_thetas_st[i - 1]) * spectrum_pre.phis_num[i];
		int spectrum_num = spectrum_pre.GetSptmNumLevel(i);
		auto& weights = spectrum_pre.weights[i];
		//if (mpipre.GetRank() >= 0) cout <<mpipre.GetRank() << " " << i << " " << spectrum_num << "数量 " << weights.size() << " " << L << " " << spectrum_pre.k_vecs[i].size() << " " << itp_local_st << endl;
		for (auto& proxy_cube : proxy_cubes) {
			for (auto& cube_index : proxy_cube.local_index) {
				VecJD3 D_vec = octree.mortoncode3d.GetGap2(proxy_cube.mtc, cubes[cube_index].mtc, i);
				//cout << D_vec.norm() / (GlobalParams.lam * 0.251) << "波长" << proxy_cube.mtc << " " << cubes[cube_index].mtc << endl;
				JD D = D_vec.norm();
				int index = -1;
				for (int j = 0; j < transfers_D.size(); ++j) {
					if ((D_vec - transfers_D[j]).norm() < 1e-5) {
						index = j;
						break;
					}
				}
				if (index == -1) {
					transfers_D.push_back(D_vec);
					VecCP transfer(spectrum_num);
					JD kD = GlobalParams.k0 * D;
					int LL = L >= int(kD) ? min(int(kD) + 3, L) : L;
					//int LL = L;
					//cout << LL << " " << kD << " " << D_vec.norm() / (GlobalParams.lam) << "波长 " << L << endl;
					if (LL < 30) {
						for (int k = 0; k < spectrum_num; ++k)
						{
							JD kdotD = (spectrum_pre.k_vecs[i][k + itp_local_st] / GlobalParams.k0).dot(D_vec / D);
							CP TL(0, 0);
							if (kdotD > 1.0) kdotD -= min_eps;
							if (kdotD < -1.0) kdotD += min_eps;
							TL = TranferF(LL, kdotD, kD, GlobalParams.k0) * weights[k];
							transfer[k] = TL;
						}
					}
					else {
						int M = 5 * LL;
						JD diff_theta = pi / M;
						vector<CP> TArray(M + 1);
						for (int k = 0; k <= M; ++k) {
							JD ct = cos(diff_theta * k);
							TArray[k] = TranferF(LL, ct, kD, GlobalParams.k0);
						}
						for (int k = 0; k < spectrum_num; ++k)
						{
							JD kdotD = (spectrum_pre.k_vecs[i][k + itp_local_st] / GlobalParams.k0).dot(D_vec / D);
							CP TL(0, 0);
							if (kdotD > 1.0) kdotD -= min_eps;
							if (kdotD < -1.0) kdotD += min_eps;
							//TL = TranferF(L, kdotD, kD);
							TL = LagrangeITF(TArray, kdotD, diff_theta, 3) * weights[k];
							transfer[k] = TL;
						}
					}
					transfers_level.push_back(transfer);
					proxy_cube.index.push_back(transfers_level.size() - 1);
					//if (i == 3) cout << mpipre.GetRank() << " " << transfers_level.size() - 1 << " " << transfer.squaredNorm() << endl;
				}
				else {
					proxy_cube.index.push_back(index);
				}
			}
		}
		transfers.push_back(transfers_level);
		mem_sum += transfers_level.size() * spectrum_num;
	}
	JD mem_all = mpiout(mem_sum, mpipre);
	if (mpipre.GetRank() == 0) cout << "Transfer done! " << MemUsed(mem_all) * 2 << "GB" << endl;
}

void MatrixPre::MemPre()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	size_t mem_size = 0;
	size_t mpi_mem_size = 0;

	for (int i = 0; i < level_num; ++i) {
		//确认当前层盒子数量
		//确认当前层代理盒子数量
		//确认当前层波谱数量
		//确认当前层插值后的波谱数量
		//确认当前层所需分配的内存大小
		auto& cubes = octree.GetCubesLevel(i);
		auto& proxy_cubes = octree.GetProxyCubesLevel(i);
		int sptm_num = spectrum_pre.weights[i].size();
		mem_size += sptm_num * cubes.size() * 2;
		if (i == level_num - 2) {
			temp_mem_ptr = (CP*)malloc((TWIP::pp * 8 * spectrum_pre.phis_num[i] + spectrum_pre.k_vecs[i + 1].size() * 2) * sizeof(CP));
			//cout << "暂时内存大小 " << TWIP::pp * 8 * spectrum_pre.phis_num[i] + spectrum_pre.k_vecs[i + 1].size() * 2 << endl;
		}
		if (i <= mpipre.Get_HSP_end()) {
			size_t max_num = 0;
			int transfer_process_num = mpipre.GetTransferProcess(i).size();
			//cout << "?" << endl;
			for (auto& local_cubes_sent_level : octree.local_cubes_sent[i]) max_num = max(max_num, local_cubes_sent_level.size());
			//cout << "??" << endl;
			for (auto& local_cubes_recv_num : octree.far_cubes_recv_num[i]) max_num = max(max_num, size_t(local_cubes_recv_num));
			int transfer_buffer_num = min(send_buffer_size + recv_buffer_size, transfer_process_num * 2);
			//cout << max_num << " 转移缓冲区数量" << transfer_buffer_num << " " << sptm_num * 2 * max_num * transfer_buffer_num << endl;
			mpi_mem_size = max(mpi_mem_size, sptm_num * (cubes.size() + max_num * transfer_buffer_num) * 2);
		}
		

		if (i < level_num - 1) {
			int itp_num = spectrum_pre.k_vecs[i + 1].size();
			auto& cubes_fa = octree.GetCubesLevel(i + 1);
			mpi_mem_size = max(mpi_mem_size, itp_num * cubes_fa.size() * 2);
		}
		//cout << "内存" << i << " " << mem_size << " " << mpi_mem_size << endl;
	}
	mem_st_ptr = (CP*)malloc(mem_size * sizeof(CP));
	mem_end_ptr = mem_st_ptr + mem_size;
	CP* temp_ptr = mem_st_ptr;
	for (int i = 0; i < level_num; i++) {
		auto& cubes = octree.GetCubesLevel(i);
		int sptm_num = spectrum_pre.GetSptmNumLevel(i);
		for (int j = 0; j < cubes.size(); j++) {
			cubes[j].sptmtheta = temp_ptr + j * sptm_num * 2;
		}
		temp_ptr += sptm_num * 2 * cubes.size();
	}
	mpi_mem_st_ptr = (CP*)malloc(mpi_mem_size * sizeof(CP));
	mpi_mem_end_ptr = mpi_mem_st_ptr + mpi_mem_size;
	JD mem_all = mpiout(JD(mem_size + mpi_mem_size), mpipre);
	if (mpipre.GetRank() == 0)cout << "分配波谱内存成功 " << MemUsed(mem_all) * 2 << " GB" << " " << MemUsed(mem_size + mpi_mem_size) << endl;
}

void MatrixPre::InterpolationProd(bool judge)
{
	int level_num = spectrum_pre.GetTopLevelNum();

	for (int i = 0; i < level_num - 1; i++) {
		int level = judge ? level_num - 2 - i : i;
		auto& cubes_fa = octree.GetCubesLevel(level + 1);
		auto& cubes_son = octree.GetCubesLevel(level);
		auto& proxy_cubes = octree.GetProxyCubesLevel(level);

		int Itp_theta_st = spectrum_pre.ip_thetas_st[level], Itp_theta_end = spectrum_pre.ip_thetas_end[level];
		int fa_phi_num = spectrum_pre.phis_num[level + 1], son_phi_num = spectrum_pre.phis_num[level];
		int fa_theta_st = spectrum_pre.sptm_thetas_st[level + 1], fa_theta_end = spectrum_pre.sptm_thetas_end[level + 1];
		int fa_sptm_num = (fa_theta_end - fa_theta_st) * fa_phi_num;

		int Itp_num = (Itp_theta_end - Itp_theta_st) * fa_phi_num;
		int son_theta_st = spectrum_pre.sptm_thetas_st[level], son_theta_end = spectrum_pre.sptm_thetas_end[level];
		int sptm_son_num = (son_theta_end - son_theta_st) * son_phi_num;

		const int exchange_st = fa_theta_end != Itp_theta_end ? fa_theta_end : Itp_theta_st;
		const int exchange_end = fa_theta_end != Itp_theta_end ? Itp_theta_end : fa_theta_st;
		const int exchange_num = (exchange_end - exchange_st) * fa_phi_num;
		const int itp_ex_st = (exchange_st - Itp_theta_st) * fa_phi_num;
		const int itp_local_st = (fa_theta_st - Itp_theta_st) * fa_phi_num;
		//cout <<"层数"<< level <<" "<<exchange_num<<endl;

		if (judge) {
			Map<VectorXcd> sptm_fa(cubes_fa[0].sptmtheta, cubes_fa.size() * 2 * fa_sptm_num);
			Map<VectorXcd> sptm_temp(mpi_mem_st_ptr, cubes_fa.size() * 2 * Itp_num);
			if (level == mpipre.Get_BP_end() - 1 && mpipre.GetProcessNum(level + 1) > 1)
			{
				/*const int cluster_num = mpipre.GetProcessNum(level + 1);
				std::vector<MPI_Request> request(cluster_num);
				for (int k = 0; k < cluster_num; k++) {
					MPI_Isend(sptm_fa.data(), cubes_fa.size() * 2 * fa_sptm_num, MPI_DOUBLE_COMPLEX, mpipre.Get_BP_com(k), cubes_fa.size() * 2 * fa_sptm_num, MPI_COMM_WORLD, &request[k]);
				}
				for (int k = 0; k < cluster_num; k++) {
					int send_num = (spectrum_pre.BP_HSP_theta_index[k + 1] - spectrum_pre.BP_HSP_theta_index[k]) * fa_phi_num * 2 * cubes_fa.size();
					int st_num = spectrum_pre.BP_HSP_theta_index[k] * fa_phi_num * 2 * cubes_fa.size();
					MPI_Recv(mpi_mem_st_ptr + st_num, send_num, MPI_DOUBLE_COMPLEX, mpipre.Get_BP_com(k), send_num, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				for (int k = 0; k < cluster_num; k++) MPI_Wait(&request[k], MPI_STATUS_IGNORE);
				Map<VectorXcd> sptm_recv(mem_st_ptr, cubes_fa.size() * 2 * Itp_num);
				sptm_recv = sptm_temp;
				for (int j = 0; j < cubes_fa.size(); j++) {
					for (int k = 0; k < cluster_num; ++k) {
						int send_num = (spectrum_pre.BP_HSP_theta_index[k + 1] - spectrum_pre.BP_HSP_theta_index[k]) * fa_phi_num;
						int st_num = spectrum_pre.BP_HSP_theta_index[k] * fa_phi_num;

						sptm_temp.segment(j * Itp_num * 2 + st_num, send_num) = sptm_recv.segment(st_num * cubes_fa.size() * 2 + j * send_num * 2, send_num);
						sptm_temp.segment(j * Itp_num * 2 + Itp_num + st_num, send_num) = sptm_recv.segment(st_num * cubes_fa.size() * 2 + j * send_num * 2 + send_num, send_num);
					}
				}*/

			}
			else if (exchange_num) {
				MPI_Sendrecv(sptm_fa.data(), cubes_fa.size() * 2 * fa_sptm_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankExchange(level), 0,
					mpi_mem_st_ptr + cubes_fa.size() * fa_sptm_num * 2, cubes_fa.size() * exchange_num * 2, MPI_DOUBLE_COMPLEX, mpipre.GetRankExchange(level), 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				Map<VectorXcd> sptm_ex(mpi_mem_st_ptr + cubes_fa.size() * fa_sptm_num * 2, cubes_fa.size() * exchange_num * 2);
				for (int j = 0; j < cubes_fa.size(); j++) {
					Map<VectorXcd> sptm_itp(mpi_mem_st_ptr + j * Itp_num * 2, Itp_num * 2);
					sptm_itp.topRows(Itp_num).segment(itp_ex_st, exchange_num) = sptm_ex.segment(j * exchange_num * 2, exchange_num);
					sptm_itp.bottomRows(Itp_num).segment(itp_ex_st, exchange_num) = sptm_ex.segment(j * exchange_num * 2 + exchange_num, exchange_num);
					sptm_itp.topRows(Itp_num).segment(itp_local_st, fa_sptm_num) = sptm_fa.segment(j * fa_sptm_num * 2, fa_sptm_num);
					sptm_itp.bottomRows(Itp_num).segment(itp_local_st, fa_sptm_num) = sptm_fa.segment(j * fa_sptm_num * 2 + fa_sptm_num, fa_sptm_num);
				}
			}
			else sptm_temp = sptm_fa;
			//if (ok)cout << mpipre.GetRank() << "??" << endl;
			auto& aSu = aSu_list[level];
			auto& aSm = aSm_list[level];
			auto& aSd = aSd_list[level];
			auto& aF = aF_list[level];
			Map<VectorXcd> spectrum_up_f(temp_mem_ptr, TWIP::pp * 2 * fa_phi_num);
			Map<VectorXcd> spectrum_down_f(temp_mem_ptr + TWIP::pp * 2 * fa_phi_num, TWIP::pp * 2 * fa_phi_num);
			Map<VectorXcd> spectrum_send_up(temp_mem_ptr + 2 * TWIP::pp * 2 * fa_phi_num, TWIP::pp * 2 * fa_phi_num);
			Map<VectorXcd> spectrum_send_down(temp_mem_ptr + 3 * TWIP::pp * 2 * fa_phi_num, TWIP::pp * 2 * fa_phi_num);

			MPI_Request request_f[4];
			MPI_Send_init(spectrum_send_up.data(), TWIP::pp * 2 * fa_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankUp(level), 0, MPI_COMM_WORLD, &request_f[0]);
			MPI_Send_init(spectrum_send_down.data(), TWIP::pp * 2 * fa_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankDown(level), 1, MPI_COMM_WORLD, &request_f[1]);
			MPI_Recv_init(spectrum_up_f.data(), TWIP::pp * 2 * fa_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankUp(level), 1, MPI_COMM_WORLD, &request_f[2]);
			MPI_Recv_init(spectrum_down_f.data(), TWIP::pp * 2 * fa_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankDown(level), 0, MPI_COMM_WORLD, &request_f[3]);

			//if (ok)cout << mpipre.GetRank() << "???" << endl;
			int son_index = 0;
			for (int j = 0; j < cubes_fa.size(); ++j)
			{
				Map<VectorXcd> fa_sptm_theta(mpi_mem_st_ptr + j * Itp_num * 2, Itp_num);
				Map<VectorXcd> fa_sptm_phi(mpi_mem_st_ptr + j * Itp_num * 2 + Itp_num, Itp_num);
				//if (ok && j == 0 && mpipre.GetRank() == 0) cout << fa_sptm_theta.topRows(131044).bottomRows(3).transpose() << "      " << Itp_num << endl;
				while (son_index < cubes_son.size() && (cubes_son[son_index].mtc >> 3) == cubes_fa[j].mtc)
				{
					VectorXcd& phase_shift = phase_shifts[level][cubes_son[son_index].mtc & 7];

					if (son_theta_st != 0) {
						spectrum_send_up.topRows(TWIP::pp * fa_phi_num).noalias() = fa_sptm_theta.topRows(TWIP::pp * fa_phi_num)
							.cwiseProduct(phase_shift.topRows(TWIP::pp * fa_phi_num).conjugate());
						spectrum_send_up.bottomRows(TWIP::pp * fa_phi_num).noalias() = fa_sptm_phi.topRows(TWIP::pp * fa_phi_num)
							.cwiseProduct(phase_shift.topRows(TWIP::pp * fa_phi_num).conjugate());
						MPI_Start(&request_f[0]);
						MPI_Start(&request_f[2]);
					}
					if (son_theta_end != spectrum_pre.thetas_num[level]) {
						spectrum_send_down.topRows(TWIP::pp * fa_phi_num).noalias() = fa_sptm_theta.bottomRows(TWIP::pp * fa_phi_num)
							.cwiseProduct(phase_shift.bottomRows(TWIP::pp * fa_phi_num).conjugate());
						spectrum_send_down.bottomRows(TWIP::pp * fa_phi_num).noalias() = fa_sptm_phi.bottomRows(TWIP::pp * fa_phi_num)
							.cwiseProduct(phase_shift.bottomRows(TWIP::pp * fa_phi_num).conjugate());
						MPI_Start(&request_f[1]);
						MPI_Start(&request_f[3]);
					}
					Map<VectorXcd> son_sptm_theta(cubes_son[son_index].sptmtheta, sptm_son_num);
					Map<VectorXcd> son_sptm_phi(cubes_son[son_index].sptmtheta + sptm_son_num, sptm_son_num);
					/*if (son_index == 0) {
						VecCP hh(sptm_son_num * 2);
						hh.topRows(sptm_son_num) = aF * (aSm * phase_shift.conjugate().cwiseProduct(fa_sptm_theta));
						hh.bottomRows(sptm_son_num) = aF * (aSm * phase_shift.conjugate().cwiseProduct(fa_sptm_phi));
						hhc(level, hh.data());
					}*/
					son_sptm_theta.noalias() += aF * (aSm * phase_shift.conjugate().cwiseProduct(fa_sptm_theta));
					son_sptm_phi.noalias() += aF * (aSm * phase_shift.conjugate().cwiseProduct(fa_sptm_phi));

					if (son_theta_st != 0) {
						MPI_Wait(&request_f[2], MPI_STATUS_IGNORE);
						son_sptm_theta.noalias() += aF * (aSu * spectrum_up_f.topRows(TWIP::pp * fa_phi_num));
						son_sptm_phi.noalias() += aF * (aSu * spectrum_up_f.bottomRows(TWIP::pp * fa_phi_num));
					}
					if (son_theta_end != spectrum_pre.thetas_num[level]) {
						MPI_Wait(&request_f[3], MPI_STATUS_IGNORE);
						son_sptm_theta.noalias() += aF * (aSd * spectrum_down_f.topRows(TWIP::pp * fa_phi_num));
						son_sptm_phi.noalias() += aF * (aSd * spectrum_down_f.bottomRows(TWIP::pp * fa_phi_num));
					}
					//if (ok && son_index == 0 && mpipre.GetRank() == 0) cout << son_sptm_theta.topRows(36864).bottomRows(3).transpose() << " " << sptm_son_num << endl;
					if (son_theta_st != 0)MPI_Wait(&request_f[0], MPI_STATUS_IGNORE);
					if (son_theta_end != spectrum_pre.thetas_num[level])MPI_Wait(&request_f[1], MPI_STATUS_IGNORE);
					++son_index;
					//cout << mpipre.GetRank() << "level:" << level << "son_index:" << son_index << " " << j << endl;
				}
			}
			//Map<VectorXcd> son_sptm(mem_st_ptr, sptm_son_num * 2 * cubes_son.size());
			//sum_level = son_sptm.squaredNorm();
			for (int i = 0; i < 4; ++i) MPI_Request_free(&request_f[i]);
		}
		else {
			auto& Fu = Fu_list[level];
			auto& Fm = Fm_list[level];
			auto& Fd = Fd_list[level];
			auto& S = S_list[level];

			Map<VectorXcd> sptm_recv_up(temp_mem_ptr, TWIP::p * 2 * son_phi_num);
			Map<VectorXcd> sptm_recv_down(temp_mem_ptr + TWIP::p * 2 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcd> sptm_send_up(temp_mem_ptr + TWIP::p * 4 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcd> sptm_send_down(temp_mem_ptr + TWIP::p * 6 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcd> itp_sptm(temp_mem_ptr + TWIP::p * 8 * son_phi_num, Itp_num * 2);

			MPI_Request request_s[4];
			MPI_Send_init(sptm_send_up.data(), TWIP::p * 2 * son_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankUp(level), 0, MPI_COMM_WORLD, &request_s[0]);
			MPI_Send_init(sptm_send_down.data(), TWIP::p * 2 * son_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankDown(level), 1, MPI_COMM_WORLD, &request_s[1]);
			MPI_Recv_init(sptm_recv_up.data(), TWIP::p * 2 * son_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankUp(level), 1, MPI_COMM_WORLD, &request_s[2]);
			MPI_Recv_init(sptm_recv_down.data(), TWIP::p * 2 * son_phi_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankDown(level), 0, MPI_COMM_WORLD, &request_s[3]);

			/*int level_t = 6;
			CP E_yan(0, 0);*/

			int son_index = 0;
			for (int j = 0; j < cubes_fa.size(); ++j) {
				itp_sptm.setZero();
				//cout << level << " " << "第几个盒子" << j << endl;
				while (son_index < cubes_son.size() && (cubes_son[son_index].mtc >> 3) == cubes_fa[j].mtc) {
					VectorXcd& phase_shift = phase_shifts[level][cubes_son[son_index].mtc & 7];

					Map<VectorXcd> sptm_son_theta(cubes_son[son_index].sptmtheta, sptm_son_num);
					Map<VectorXcd> sptm_son_phi(cubes_son[son_index].sptmtheta + sptm_son_num, sptm_son_num);
					//if (ok)cout << mpipre.GetRank() << " ?" << fa_index << " " <<j<< endl;
					if (son_theta_st != 0) {
						sptm_send_up.topRows(TWIP::p * son_phi_num) = sptm_son_theta.topRows(TWIP::p * son_phi_num);
						sptm_send_up.bottomRows(TWIP::p * son_phi_num) = sptm_son_phi.topRows(TWIP::p * son_phi_num);
						MPI_Start(&request_s[0]), MPI_Start(&request_s[2]);
					}
					if (son_theta_end != spectrum_pre.thetas_num[level]) {
						sptm_send_down.topRows(TWIP::p * son_phi_num) = sptm_son_theta.bottomRows(TWIP::p * son_phi_num);
						sptm_send_down.bottomRows(TWIP::p * son_phi_num) = sptm_son_phi.bottomRows(TWIP::p * son_phi_num);
						MPI_Start(&request_s[1]), MPI_Start(&request_s[3]);
					}
					itp_sptm.topRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * (Fm * sptm_son_theta));
					itp_sptm.bottomRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * (Fm * sptm_son_phi));
					//if (ok)cout << mpipre.GetRank() << " ??" << mpipre.GetRankDown(level) << " " << j << " "<< sptm_send_down.topRows(2).transpose() <<" "<< mpipre.GetRankUp(level)<<endl;
					if (son_theta_st != 0) {
						MPI_Wait(&request_s[2], MPI_STATUS_IGNORE);
						itp_sptm.topRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * (Fu * sptm_recv_up.topRows(TWIP::p * son_phi_num)));
						itp_sptm.bottomRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * (Fu * sptm_recv_up.bottomRows(TWIP::p * son_phi_num)));
					}

					if (son_theta_end != spectrum_pre.thetas_num[level]) {
						MPI_Wait(&request_s[3], MPI_STATUS_IGNORE);
						itp_sptm.topRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * (Fd * sptm_recv_down.topRows(TWIP::p * son_phi_num)));
						itp_sptm.bottomRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * (Fd * sptm_recv_down.bottomRows(TWIP::p * son_phi_num)));
					}
					if (son_theta_st != 0) MPI_Wait(&request_s[0], MPI_STATUS_IGNORE);
					if (son_theta_end != spectrum_pre.thetas_num[level]) MPI_Wait(&request_s[1], MPI_STATUS_IGNORE);
					++son_index;
				}
				Map<VectorXcd> sptm_fa(mpi_mem_st_ptr + j * 2 * fa_sptm_num, fa_sptm_num * 2);
				if (level == mpipre.Get_BP_end() - 1 && mpipre.GetProcessNum(level + 1) > 1)
				{
					/*const int cluster_num = mpipre.GetProcessNum(level + 1);
					for (int k = 0; k < cluster_num; k++) {
						int send_num = (spectrum_pre.BP_HSP_theta_index[k + 1] - spectrum_pre.BP_HSP_theta_index[k]) * fa_phi_num;
						int st_num = spectrum_pre.BP_HSP_theta_index[k] * fa_phi_num;
						Map<VectorXcd> sptm_send(mpi_mem_st_ptr + cubes_fa.size() * 2 * st_num + j * send_num * 2, send_num * 2);
						sptm_send.topRows(send_num) = itp_sptm.topRows(Itp_num).segment(st_num, send_num);
						sptm_send.bottomRows(send_num) = itp_sptm.bottomRows(Itp_num).segment(st_num, send_num);
					}*/
					//cout << mpipre.GetRank() << "????" << endl;
				}
				else if (exchange_num) {
					Map<VectorXcd> sptm_exchange(mpi_mem_st_ptr + cubes_fa.size() * 2 * fa_sptm_num + j * exchange_num * 2, exchange_num * 2);
					sptm_exchange.topRows(exchange_num).noalias() = itp_sptm.topRows(Itp_num).segment(itp_ex_st, exchange_num);
					sptm_exchange.bottomRows(exchange_num).noalias() = itp_sptm.bottomRows(Itp_num).segment(itp_ex_st, exchange_num);
					sptm_fa.topRows(fa_sptm_num).noalias() = itp_sptm.topRows(Itp_num).segment(itp_local_st, fa_sptm_num);
					sptm_fa.bottomRows(fa_sptm_num).noalias() = itp_sptm.bottomRows(Itp_num).segment(itp_local_st, fa_sptm_num);
					//cout << mpipre.GetRank() << "?????" << endl;
				}
				else sptm_fa.noalias() = itp_sptm;
				/*if(level==2)
				{
					JD sum = sptm_fa.topRows(fa_sptm_num).squaredNorm();
					JD all_sum = 0;
					MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
					if (mpipre.GetRank() == 0) cout << level << "  " << j << " 父亲总范数 " << all_sum << endl;
					if(mpipre.GetSize()>1) cout << mpipre.GetRank() << " 本地范数 " << sptm_fa.squaredNorm() << endl;
				}*/
			}
			//cout<<mpipre.GetRank()<<" ???"<<endl;
			Map<VectorXcd> sptm_fa(cubes_fa[0].sptmtheta, cubes_fa.size() * 2 * fa_sptm_num);
			if (level == mpipre.Get_BP_end() - 1 && mpipre.GetProcessNum(level + 1) > 1)
			{
				/*const int cluster_num = mpipre.GetProcessNum(level + 1);
				std::vector<MPI_Request> request(cluster_num);
				for (int k = 0; k < cluster_num; k++) {
					int send_num = (spectrum_pre.BP_HSP_theta_index[k + 1] - spectrum_pre.BP_HSP_theta_index[k]) * fa_phi_num * 2;
					int st_num = spectrum_pre.BP_HSP_theta_index[k] * fa_phi_num * 2;
					MPI_Isend(mpi_mem_st_ptr + cubes_fa.size() * st_num, cubes_fa.size() * send_num, MPI_DOUBLE_COMPLEX, mpipre.Get_BP_com(k), 0, MPI_COMM_WORLD, &request[k]);
				}
				Map<VectorXcd> sptm_recv(mem_st_ptr + cubes_fa.size() * 2 * fa_sptm_num, cubes_fa.size() * 2 * fa_sptm_num);
				for (int k = 0; k < cluster_num; k++) {
					MPI_Recv(sptm_recv.data(), cubes_fa.size() * 2 * fa_sptm_num, MPI_DOUBLE_COMPLEX, mpipre.Get_BP_com(k), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (k == 0) sptm_fa.noalias() = sptm_recv;
					else sptm_fa.noalias() += sptm_recv;
				}
				for (int k = 0; k < cluster_num; k++) MPI_Wait(&request[k], MPI_STATUS_IGNORE);*/
			}
			else if (exchange_num) {
				//if(mpipre.GetRank() == 0)cout<<level<<" ???"<<endl;
				MPI_Sendrecv(mpi_mem_st_ptr + cubes_fa.size() * fa_sptm_num * 2, cubes_fa.size() * exchange_num * 2, MPI_DOUBLE_COMPLEX, mpipre.GetRankExchange(level), 0,
					sptm_fa.data(), cubes_fa.size() * 2 * fa_sptm_num, MPI_DOUBLE_COMPLEX, mpipre.GetRankExchange(level), 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				Map<VectorXcd> sptm_local(mpi_mem_st_ptr, cubes_fa.size() * fa_sptm_num * 2);
				sptm_fa += sptm_local;
			}
			else {
				Map<VectorXcd> sptm(mpi_mem_st_ptr, cubes_fa.size() * 2 * fa_sptm_num);
				sptm_fa.noalias() = sptm;
			}
			
			/*if (mpipre.GetSize() == 2 && level == 2) {
				cout << level << " 部分范数 " << sptm_fa.topRows(fa_sptm_num).squaredNorm() << " " << fa_sptm_num << endl;
			}
			if (mpipre.GetSize() == 1 && level == 2) {
				cout << level << " 部分范数 " << sptm_fa.topRows(fa_sptm_num).topRows(1156).squaredNorm() << " " << sptm_fa.topRows(fa_sptm_num).bottomRows(1088).squaredNorm() << " "<<fa_sptm_num << endl;
			}*/
			for (int j = 0; j < 4; ++j) MPI_Request_free(&request_s[j]);
			/*{
				Map<VecCP> sptm_son(cubes_son[0].sptmtheta, sptm_son_num * 2 * cubes_son.size());
				JD sum = sptm_son.squaredNorm();
				JD all_sum = 0;
				MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				if (mpipre.GetRank() == 0) cout <<level<< " 总范数 " << all_sum << endl;
				
			}*/
			StartTransfer(level);
			/*{
				Map<VecCP> sptm_son(cubes_son[0].sptmtheta, sptm_son_num * 2 * cubes_son.size());
				JD sum = sptm_son.squaredNorm();
				JD all_sum = 0;
				MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				if (mpipre.GetRank() == 0) cout << " 总范数 " << all_sum << endl;
			}*/
		}
	}
	//cout << "st" << endl;
	if (!judge) {
		/*{
			auto& cubes_son = octree.GetCubesLevel(level_num - 1);
			int sptm_son_num = spectrum_pre.GetSptmNumLevel(level_num - 1);
			Map<VecCP> sptm_son(cubes_son[0].sptmtheta, sptm_son_num * 2 * cubes_son.size());
			JD sum = sptm_son.squaredNorm();
			JD all_sum = 0;
			MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if (mpipre.GetRank() == 0) cout << level_num - 1 << " 总范数 " << all_sum << endl;
		}*/
		
		StartTransfer(level_num - 1);
		/*auto& cubes_son = octree.GetCubesLevel(level_num - 1);
		int sptm_son_num = spectrum_pre.GetSptmNumLevel(level_num - 1);
		Map<VecCP> sptm_son(cubes_son[0].sptmtheta, sptm_son_num * 2 * cubes_son.size());
		JD sum = sptm_son.squaredNorm();
		JD all_sum = 0;
		MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (mpipre.GetRank() == 0) cout << " 总范数 " << all_sum << endl;*/
	}
}

void MatrixPre::StartTransfer(const int level)
{
	int sptm_num = spectrum_pre.GetSptmNumLevel(level);
	auto& cubes = octree.GetCubesLevel(level);
	std::vector<int>& transfer_process = mpipre.GetTransferProcess(level);
	int m0 = lower_bound(transfer_process.begin(), transfer_process.end(), mpipre.GetRank()) - transfer_process.begin();
	//cout << mpipre.GetRank() << " m0=" << m0 << " " << level << " "<< transfer_process.size()<< endl;
	int process_num = transfer_process.size();
	//将本地波谱转移给发送数组
	CP* const transfer_send_st = mpi_mem_st_ptr + sptm_num * 2 * cubes.size();
	size_t rest_mem = mpi_mem_end_ptr - transfer_send_st;
	size_t transfer_buffer = rest_mem / (send_buffer_size + recv_buffer_size);
	CP* const transfer_recv_st = transfer_send_st + transfer_buffer * send_buffer_size;
	std::queue<MPI_Request> request_send;
	std::queue<MPI_Request> request_recv;
	int recv_num = 0, send_num = 0;
	auto& send_cubes_index = octree.local_cubes_sent[level];
	auto& recv_cubes_num = octree.far_cubes_recv_num[level];
	auto& proxy_cubes = octree.GetProxyCubesLevel(level);
	Map<VecCP> cubes_temp(mpi_mem_st_ptr, sptm_num * 2 * cubes.size());
	Map<VecCP> cubes_sptm(cubes[0].sptmtheta, sptm_num * 2 * cubes.size());
	cubes_temp = cubes_sptm;
	cubes_sptm.setZero();
	//非阻塞发送和接收
	while (recv_num < process_num) {
		//将send_index对应盒子的波谱填到发送缓冲区中（用两个指针判断有没有填满），将缓冲区中的数据发送出去
		//进行非阻塞接收（用两个指针判断有没有接）
		//检查有没有接收到，如果接收到就进行转移,同时recv_index++,移动指针
		//检查有没有发送完
		//cout << "?" << endl;
		CP* send_st = transfer_send_st;
		CP* recv_st = transfer_recv_st;
		//cout<<mpipre.GetRank()<<"  "<< send_num <<" "<< request_send.size()<<endl;
		while (send_num < process_num && request_send.size() < send_buffer_size) {
			int send_index = (send_num + m0 + 1) % process_num;
			int send_rank = transfer_process[send_index];
			auto& cubes_index = send_cubes_index[send_index];
			//cout <<mpipre.GetRank() << "发送给其他进程的盒子数量：" << cubes_index.size() << " " << send_rank << " " << sptm_num * 2 * cubes_index.size() << endl;
			if (cubes_index.size() == 0) {
				++send_num;
				continue;
			}
			for (int j = 0; j < cubes_index.size(); j++) {
				Map<VecCP> sptm_send(send_st + sptm_num * 2 * j, sptm_num * 2);
				Map<VecCP> sptm_cube(mpi_mem_st_ptr + sptm_num * 2 * cubes_index[j], sptm_num * 2);
				//cout << j <<endl;
				sptm_send = sptm_cube;
			}
			Map<VecCP> sss(send_st, sptm_num * 2 * cubes_index.size());
			//cout <<mpipre.GetRank() << " 发送范数：" << sss.squaredNorm() << endl;
			request_send.push(MPI_Request());
			//cout <<mpipre.GetRank() << "  " <<send_rank << "发 " << cubes_index.size() << " "<< sptm_num * 2 * cubes_index.size()<<endl;
			MPI_Isend(send_st, sptm_num * 2 * cubes_index.size(), MPI_DOUBLE_COMPLEX, send_rank, cubes_index.size(), MPI_COMM_WORLD, &request_send.back());
			send_st += transfer_buffer;
			++send_num;
		}
		//cout << "??" << endl;
		while (request_recv.size() < recv_buffer_size && recv_num < process_num) {
			int recv_index = (m0 - recv_num - 1 + process_num) % process_num;
			int recv_rank = transfer_process[recv_index];
			if (recv_cubes_num[recv_rank] == 0) {
				++recv_num;
				continue;
			}
			request_recv.push(MPI_Request());
			//cout <<mpipre.GetRank() << "  " << recv_rank << "接 " << recv_cubes_num[recv_rank] << " " << transfer_buffer << endl;
			MPI_Irecv(recv_st, transfer_buffer, MPI_DOUBLE_COMPLEX, recv_rank, recv_cubes_num[recv_rank], MPI_COMM_WORLD, &request_recv.back());
			recv_st += transfer_buffer;
			++recv_num;
		}
		//cout << "???" << " "<< request_recv.empty()<<endl;
		recv_st = transfer_recv_st;
		MPI_Status status;
		//if (level == 2 && mpipre.GetRank() == 1)cout << "gg??? " << request_recv.empty()<< endl;
		while (!request_recv.empty()) {
			//if (level == 2 && mpipre.GetRank() == 1)cout << "????? " << request_recv.empty() << endl;
			MPI_Wait(&request_recv.front(), &status);
			//if (level == 2 && mpipre.GetRank() == 1)cout << "?????? " << request_recv.empty() << endl;
			int source_rank = status.MPI_SOURCE;
			//if (level == 2 && mpipre.GetRank() == 1)cout << "? " << octree.far_cubes_process_index[level].size() << endl;
			int st = octree.far_cubes_process_index[level][source_rank], ed = octree.far_cubes_process_index[level][source_rank + 1];
			//if (level == 2 && mpipre.GetRank() == 1)cout << "???????? " << request_recv.empty() << endl;
			Map<VecCP> rrr(recv_st, recv_cubes_num[source_rank] * sptm_num * 2);
			//cout << mpipre.GetRank() << "收到来自" << source_rank << "的" << recv_cubes_num[source_rank] << "个盒子 " << rrr.squaredNorm() <<endl;
			for (int i = st; i < ed ; ++i) {
				//cout << "需要被转移的本地盒子数量：" << proxy_cubes[i].local_index.size() << endl;
				Map<VecCP> sptm_recv(recv_st + sptm_num * 2 * (i - st), sptm_num * 2);
				for (int j = 0; j < proxy_cubes[i].local_index.size(); j++) {
					int local_cube_index = proxy_cubes[i].local_index[j];
					int transfer_index = proxy_cubes[i].index[j];
					//cout << "代理：" << proxy_cubes[i].mtc << "本地： " << cubes[local_cube_index].mtc << " " << transfer_index << endl;
					Map<VecCP> sptm_cube(cubes[local_cube_index].sptmtheta, sptm_num * 2);
					//cout << sptm_cube.size() << endl;
					//cout << sptm_recv.size() << endl;
					//cout << transfers[level][transfer_index].size() << endl;
					//if (i == st) cout <<mpipre.GetRank()<<" "<< sptm_cube.squaredNorm() << "转移范数测试 " << sptm_recv.squaredNorm() << " " << transfers[level][transfer_index].squaredNorm() << " " << transfer_index << endl;
					sptm_cube.topRows(sptm_num).noalias() += sptm_recv.topRows(sptm_num).cwiseProduct(transfers[level][transfer_index]);
					sptm_cube.bottomRows(sptm_num).noalias() += sptm_recv.bottomRows(sptm_num).cwiseProduct(transfers[level][transfer_index]);
					//if(i == st) cout << "转移后范数" << i << " " << sptm_cube.squaredNorm() << endl;
					/*if (((i == 55 && level == 1) || (i == 204 && level == 0)) && local_cube_index == 0) {
						hhc(level, sptm_cube.data());
					}*/
				}
			}
			recv_st += transfer_buffer;
			request_recv.pop();
		}
		//cout << mpipre.GetRank() << " ???? " << endl;
		while (!request_send.empty()) {
			MPI_Wait(&request_send.front(), MPI_STATUS_IGNORE);
			request_send.pop();
		}
	}
	//cout << "转移后范数" << level << " " << cubes_sptm.squaredNorm() << endl;
}

Eigen::Vector2cd MatrixPre::GetRwgEFarField(const JD theta, const JD phi, const Eigen::VectorXcd& J)
{
	auto& cubes_floor = octree.GetCubesLevel(0);
	auto floor_cubes_num = cubes_floor.size();
	JD ct = cos(theta * rad), st = sin(theta * rad), cp = cos(phi * rad), sp = sin(phi * rad);
	Eigen::Vector3d r_unit{ st * cp ,st * sp,ct };

	Eigen::MatrixXcd c = Eigen::MatrixXcd::Zero(2, J.size());
	Matrix<JD, 2, 3> T1;
	T1 << ct * cp, ct* sp, -st,
		-sp, cp, 0;
	//cout << 0 << endl;
	//遍历每一个盒子
	int edge_index = 0;
	const RWG& rwg = octree.rwg;
	for (auto i = 0; i < floor_cubes_num; ++i)
	{
		int cube_rwgs_num = octree.cube_rwgs_num[i];
		//遍历每一个方向（行）
		for (auto j = 0; j < cube_rwgs_num; ++j)
		{
			const Eigen::Matrix<JD, 3, GLPN2>& Jj = rwg.J_GL[edge_index];
			Vector3cd e = Vector3cd::Zero();
			for (auto m = 0; m < 2; ++m) {
				double pmm = m ? -1 : 1;
				for (int p = 0; p < GLPN; ++p) {
					Vector3d Jm = pmm * (Jj.col(m * GLPN + p) - rwg.points[rwg.vertex_edges[edge_index][m]]);
					JD inrpr = r_unit.dot(Jj.col(m * GLPN + p));
					JD ct = cos(GlobalParams.k0 * inrpr), st = sin(GlobalParams.k0 * inrpr);
					//Vector3d n_unit = rwg.tri_normal[rwg.edges[edge_index][m]];
					//Vector3d mc = 1.0 / Zf * r_unit.normalized().cross(n_unit.cross(Jm));
					Vector3cd ej = GL_points[p % GLPN][0] * rwg.edges_length[edge_index] * (Jm ) * CP(ct, st);
					e += CP(0.0, -Zf * GlobalParams.k0 / (4.0 * pi)) * ej;
				}
			}
			Vector2cd e_r = T1 * e * 0.5;
			c.block(0, edge_index++, 2, 1) = e_r;
		}
	}
	return c * J;
}

Eigen::Vector3cd MatrixPre::GetRwgENearField(const VecJD3 r, const Eigen::VectorXcd& J)
{
	auto& cubes_floor = octree.GetCubesLevel(0);
	const RWG& rwg = octree.rwg;
	Eigen::Vector3cd ej = Eigen::Vector3cd::Zero();
	Eigen::Vector3cd e = Eigen::Vector3cd::Zero();
	const CP C1 = CP(0, -1.0 / (2.0 * pi * GlobalParams.freq * epsilon0 * 4.0 * pi));

	std::vector<int> rwg_index;

	auto& proxy_cube0 = octree.GetProxyCubesLevel(0)[0];
	for (auto& local_cube : proxy_cube0.local_index) {
		int st = octree.cube_rwgs_dif[local_cube];
		int ed = octree.cube_rwgs_num[local_cube] + st;
		for (int i = st; i < ed; ++i) rwg_index.push_back(i);
	}
	int level =1;
	auto& cubes = octree.GetCubesLevel(level);
	auto& proxy_cube1 = octree.GetProxyCubesLevel(level)[0];
	for (int i = 0; i < rwg.edges.size(); i++) {
		const VecJD3 point = rwg.GetEdgeCenter(i);
		size_t mtc = (octree.mortoncode3d.GetMortonCode(point) >> (3 * level));
		for (auto& index : proxy_cube1.local_index) {
			if (mtc == cubes[index].mtc) {
				rwg_index.push_back(i);
			}
		}
	}
	//cout <<"数量："<< rwg_index.size() << endl;
	//for(int i=0;i<rwg.edges.size();++i)
	for (auto& i : rwg_index) {
		const Eigen::Matrix<JD, 3, GLPN2>& Ji = rwg.J_GL[i];
		for (int m = 0; m < 2; ++m) {
			double pmm = m ? -1 : 1;
			for (int p = 0; p < GLPN; ++p) {
				VecJD3 Jm = pmm * (Ji.col(m * GLPN + p) - rwg.points[rwg.vertex_edges[i][m]]);
				CP jx = Jm[0], jy = Jm[1], jz = Jm[2];
				VecJD3 R_v = r - rwg.points[rwg.vertex_edges[i][m]];
				JD R = R_v.norm();
				JD ct = cos(-GlobalParams.k0 * R), st = sin(-GlobalParams.k0 * R);
				CP G1 = CP(1, GlobalParams.k0 * R) * CP(ct, st) / (R * R * R);
				CP G2 = CP(3 - GlobalParams.k0 * GlobalParams.k0 * R * R, 3 * GlobalParams.k0 * R) * CP(ct, st) / (R * R * R * R * R);

				ej(0) = J(i) * (2.0 * G1 * jx + G2 * (R_v(0) * (R_v(2) * jz + R_v(1) * jy) - jx * (R_v(1) * R_v(1) + R_v(2) * R_v(2))));
				ej(1) = J(i) * (2.0 * G1 * jy + G2 * (R_v(1) * (R_v(0) * jx + R_v(2) * jz) - jy * (R_v(2) * R_v(2) + R_v(0) * R_v(0))));
				ej(2) = J(i) * (2.0 * G1 * jz + G2 * (R_v(2) * (R_v(1) * jy + R_v(0) * jx) - jz * (R_v(0) * R_v(0) + R_v(1) * R_v(1))));

				e += C1 * rwg.edges_length[i] * GL_points[p % GLPN][0] * ej * 0.5;
			}
		}
	}
	return e;
}