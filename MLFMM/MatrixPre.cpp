#include "MatrixPre.h"

using namespace std;
using namespace Eigen;

void Interpolation::GetNearNb()
{
	octree.rwg.initial(spectrum_pre.freq);
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	auto& near_cubes = octree.near_cubes;
	Zinv_list.reserve(cube_floor.size());
	for (int i = 0; i < cube_floor.size(); i++) {
		int st = octree.cube_rwgs_dif[i], ed = st + octree.cube_rwgs_num[i];
		MatrixXcf Zs(octree.cube_rwgs_num[i], octree.cube_rwgs_num[i]);
		for (int j = st; j < ed; j++) {
			for (int k = j; k < ed; k++) {
				Zs(j - st, k - st) = octree.rwg.GetZij(j, k);
				Zs(k - st, j - st) = Zs(j - st, k - st);
			}
		}
		Eigen::PartialPivLU<Eigen::MatrixXcf > Zinv(Zs);
		Zinv_list.push_back(Zinv);
	}

	int matrix_size = 0;
	for (auto& near_cube : near_cubes) matrix_size += near_cube.local_index.size();
	near_matrix_list.reserve(matrix_size);


	int proxy_rwgs_st = octree.cube_rwgs_dif[cube_floor.size() - 1] + octree.cube_rwgs_num[cube_floor.size() - 1];
	for (int k = 0; k < near_cubes.size(); k++) {
		int rwgs_st = proxy_rwgs_st, rwgs_end = proxy_rwgs_st + octree.near_cube_rwgs_num[k], num = octree.near_cube_rwgs_num[k];
		if (octree.near_cube_rwgs_num[k] == 0) {
			int m0 = lower_bound(cube_floor.begin(), cube_floor.end(), near_cubes[k].mtc, [](const Cube& cube, const size_t& mtc) {return cube.mtc < mtc; }) - cube_floor.begin();
			rwgs_st = octree.cube_rwgs_dif[m0];
			rwgs_end = octree.cube_rwgs_dif[m0] + octree.cube_rwgs_num[m0];
			num = octree.cube_rwgs_num[m0];
		}
		for (auto& index : near_cubes[k].local_index) {
			int local_rwgs_end = octree.cube_rwgs_dif[index] + octree.cube_rwgs_num[index];
			MatrixXcf Zn(octree.cube_rwgs_num[index], num);
			for (int i = octree.cube_rwgs_dif[index]; i < local_rwgs_end; i++) {
				for (int j = rwgs_st; j < rwgs_end; j++) {
					Zn(i - octree.cube_rwgs_dif[index], j - rwgs_st) = octree.rwg.GetZij(i, j);
				}
			}
			near_matrix_list.push_back(Zn);
			near_cubes[k].index.push_back(near_matrix_list.size() - 1);
		}
		proxy_rwgs_st += octree.near_cube_rwgs_num[k];
	}
}

void Interpolation::NearProd(Eigen::VectorXcf& x, Eigen::VectorXcf& b)
{
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
		if (cubes_sent.empty()) continue;
		int shift = 0;
		for (auto& j : cubes_sent) {
			Map<VectorXcf> x_sent(mpi_mem_st_ptr + shift, octree.cube_rwgs_num[j]);
			x_sent = x.segment(octree.cube_rwgs_dif[j], octree.cube_rwgs_num[j]);
			shift += octree.cube_rwgs_num[j];
		}
		send_requests.push(MPI_Request());
		MPI_Isend(st, shift , MPI_COMPLEX, m0, 0, MPI_COMM_WORLD, &send_requests.back());
		st += shift;
	}
	CP* st_temp = st;
	for (int i = 0; i < mpipre.GetSize(); i++) {
		int m0 = (rank - i - 1 + mpipre.GetSize()) % mpipre.GetSize();
		if(octree.near_cubes_recv_num[i] == 0) continue;
		recv_requests.push(MPI_Request());
		MPI_Irecv(st, octree.near_cubes_recv_num[i], MPI_COMPLEX, m0, 0, MPI_COMM_WORLD, &recv_requests.back());
		st += octree.near_cubes_recv_num[i];
	}
	cout << "通信数量" << mpipre.GetSize() << " " << send_requests.size() << " " << recv_requests.size() << endl;
	for (int i = 0; i < mpipre.GetSize(); i++) {
		int m0 = (rank - i - 1 + mpipre.GetSize()) % mpipre.GetSize();
		if (octree.near_cubes_recv_num[i] == 0) continue;
		MPI_Wait(&recv_requests.front(), MPI_STATUS_IGNORE);
		int near_process_st = octree.near_cubes_process_index[m0];
		while (near_cubes[near_process_st].process_index == m0) {
			Map<VectorXcf> x_recv(st_temp, octree.near_cube_rwgs_num[m0]);
			for (int j = 0; j < near_cubes[near_process_st].index.size(); j++) {
				int cube_index = near_cubes[near_process_st].local_index[j];
				int rwgs_st = octree.cube_rwgs_dif[cube_index];
				b.segment(rwgs_st, octree.near_cube_rwgs_num[cube_index]).noalias() += near_matrix_list[near_cubes[near_process_st].index[j]] * x_recv;
			}
			st_temp += octree.near_cube_rwgs_num[m0];
		}
	}
}

void Aggregation::GetAggregations()
{
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	int level_num = spectrum_pre.level_num;
	int spectrum_num = spectrum_pre.k_vecs[level_num].size();
	//cout <<"聚合"<< spectrum_num << endl;
	aggregations.reserve(cube_floor.size());
	const RWG& rwg = octree.rwg;
	int edge_index = 0;
	for (int i = 0; i < cube_floor.size(); i++) {
		auto& cube = cube_floor[i];
		Vector3f point = octree.mortoncode3d.GetPoint(cube.mtc, 0);
		MatrixXcf aggregation(spectrum_num * 2, octree.cube_rwgs_num[i]);
		for (int row = 0; row < spectrum_num; row++) {
			for (int j = 0; j < octree.cube_rwgs_num[i]; j++) {
				const Eigen::Matrix<float, 3, 6>& Jj = rwg.J_GL[edge_index];
				Vector3cf e = Vector3cf::Zero();
				for (auto m = 0; m < 2; ++m)
				{
					float pmm = m ? -1.0 : 1.0;
					for (int p = 0; p < GLPN; p++) {
						Vector3f Jm = pmm * (Jj.col(m * 3 + p) - rwg.points[rwg.vertex_edges[edge_index][m]]);
						float inrpr = (spectrum_pre.k_vecs[level_num][row]).dot(Jj.col(m * 3 + p) - point);
						float ct = cos(inrpr), st = sin(inrpr);
						//Vector3f n_unit = rwg.tri_normal[rwg.edges[edge_index][m]];
						//Vector3f mc = 1.0 / Zf * (spectrum_pre.KUnit(spectrum_pre.LevelNum(), row)).normalized().cross(n_unit.cross(Jm));
						Vector3cf ej = GL_points[p][0] * rwg.edges_length[edge_index] * (Jm) * CP(ct, st);
						e += ej;// +CPD(0.0, k_unit[0].norm() / (4.0 * pi)) * mj;
					}

				}
				aggregation.block(row * 2, j, 2, 1).noalias() = spectrum_pre.TCtoS[row] * e * 0.5;
				++edge_index;
			}
		}
		aggregations.push_back(aggregation);
	}

	TWIP twip(spectrum_pre.thetas[0], spectrum_pre.thetas[level_num], spectrum_pre.phis_num[0], spectrum_pre.phis_num[level_num]);
	twip.GetMatrix(0, spectrum_pre.thetas_num[0], 0, spectrum_pre.thetas_num[level_num]);
	F = twip.First;
	S = twip.Second;
	aF = twip.First.transpose();
	aS = twip.Second.transpose();

	if (MPIpre().GetRank() == 0)cout << "Aggregation Done!" << endl;
}

void Aggregation::AggregationProd(Eigen::VectorXcf& x, bool judge)
{
	std::vector<Cube>& cube_floor = octree.GetCubesLevel(0);
	int spectrum_num = spectrum_pre.GetSpectrumNumLevel(0);
	//cout << "波谱数量" << spectrum_num1 << " "<< spectrum_num2 << " "<< spectrum_pre.GetInterpolation().rows()<<" "<< spectrum_pre.GetInterpolation().cols()<< endl;
	int offset = 0;
	double sum = 0;
	for (auto i = 0; i < cube_floor.size(); ++i)
	{
		int cube_rwgs_num = octree.cube_rwgs_num[i];
		auto& aggregation = aggregations[i];
		if (judge) {
			/*Map<VectorXcd> spectrum(cube_floor[i].sptm_ptr, spectrum_num1 * 2);
			x.segment(offset, cube_rwgs_num).noalias() = aggregation.adjoint() * spectrum;
			cube_floor[i].sptm_ptr = nullptr;*/
		}
		else {
			Map<VectorXcf> spectrum_theta(cube_floor[i].sptmtheta, spectrum_num);
			Map<VectorXcf> spectrum_phi(cube_floor[i].sptmtheta, spectrum_num);
			spectrum_theta.noalias() = S * (F * (aggregation.topRows(spectrum_num) * x.segment(offset, cube_rwgs_num)));
			spectrum_phi.noalias() = S * (F * (aggregation.bottomRows(spectrum_num) * x.segment(offset, cube_rwgs_num)));
			//for (int j = hh_theta_num / 2; j < hh_theta_num; j++) {
			//	for (int k = (hh_theta_num % 2 && j == hh_theta_num / 2) ? hh_phi_num / 2 : 0; k < hh_phi_num; k++) {
			//		int index = (j * hh_phi_num + k) * 2;
			//		int hh = ((hh_theta_num - j - 1) * hh_phi_num + (k + hh_phi_num / 2) % hh_phi_num) * 2;
			//		//if (i == 0)cout << j << " " << k << " " << index << "  " << hh << " " << hh_theta_num % 2 <<" "<< (j == hh_theta_num / 2) << " "<< hh_theta_num / 2<<endl;
			//		spectrum.segment(index, 2).noalias() = aggregation.middleRows(hh, 2).conjugate() * x_seg;
			//	}
			//}
			/*VectorXcd spectrum(spectrum_num1 * 2);
			spectrum.segment(0, spectrum_num1 * 2) = aggregation * x.segment(offset, cube_rwgs_num);
			(*cube_floor[i].spectrum_ptr)(Eigen::seq(0, Eigen::last, 2)) = spectrum_pre.GetInterpolation() * spectrum(Eigen::seq(0, Eigen::last, 2));
			(*cube_floor[i].spectrum_ptr)(Eigen::seq(1, Eigen::last, 2)) = spectrum_pre.GetInterpolation() * spectrum(Eigen::seq(1, Eigen::last, 2));*/
			//sum += spectrum.squaredNorm();
		}
		offset += cube_rwgs_num;
	}
}

void Interpolation::GetInterpolations()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	Fu_list.reserve(level_num);
	Fm_list.reserve(level_num);
	Fd_list.reserve(level_num);
	aFu_list.reserve(level_num );
	aFm_list.reserve(level_num);
	aFd_list.reserve(level_num);
	S_list.reserve(level_num );
	aS_list.reserve(level_num );
	for (int i = 0; i < level_num - 1; i++) {
		int fa_theta_st = spectrum_pre.ip_thetas_st[i], fa_theta_end = spectrum_pre.ip_thetas_end[i];
		int son_theta_st = spectrum_pre.sptm_thetas_st[i], son_theta_end = spectrum_pre.sptm_thetas_end[i];
		int fa_phi_num = spectrum_pre.phis_num[i + 1], son_phi_num = spectrum_pre.phis_num[i];

		int son_shift = fa_theta_st ? TWIP::p * fa_phi_num : 0;
		int fa_shift = fa_theta_st ? TWIP::pp * fa_phi_num : 0;
		int actual_fa_spectrum_num = (fa_theta_end - fa_theta_st) * fa_phi_num;
		int actual_temp_spectrum_num = (fa_theta_end - fa_theta_st) * son_phi_num;
		int actual_son_spectrum_num = (son_theta_end - son_theta_st) * son_phi_num;

		TWIP twip(spectrum_pre.thetas[i + 1], spectrum_pre.thetas[i], spectrum_pre.phis_num[i + 1], spectrum_pre.phis_num[i]);
		twip.GetMatrix(fa_theta_st, fa_theta_end, son_theta_st, son_theta_end);

		Fu_list.push_back(twip.First.middleRows(fa_shift, actual_temp_spectrum_num).leftCols(TWIP::p * son_phi_num));
		Fm_list.push_back(twip.First.middleRows(fa_shift, actual_temp_spectrum_num).middleCols(son_shift, actual_son_spectrum_num));
		Fd_list.push_back(twip.First.middleRows(fa_shift, actual_temp_spectrum_num).rightCols(TWIP::p * son_phi_num));
		S_list.push_back(twip.Second.block(fa_shift, fa_shift, actual_fa_spectrum_num, actual_temp_spectrum_num));

		aFu_list.push_back(twip.First.transpose().middleRows(son_shift, actual_son_spectrum_num).leftCols(TWIP::pp * fa_phi_num));
		aFm_list.push_back(twip.First.transpose().middleRows(son_shift, actual_son_spectrum_num).middleCols(fa_shift, actual_fa_spectrum_num));
		aFd_list.push_back(twip.First.transpose().middleRows(son_shift, actual_son_spectrum_num).rightCols(TWIP::pp * fa_phi_num));
		aS_list.push_back(twip.Second.transpose());
	}
}

void Interpolation::GetPhaseShifts()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	phase_shifts.reserve(level_num);

	for (int i = 0; i < level_num - 1; ++i)
	{
		unordered_map<int, int> phase_shift_map_now;
		vector<VectorXcd> phase_shift_now;
		phase_shift_now.resize(8);

		int spectrum_father_num = (spectrum_pre.ip_thetas_end[i] - spectrum_pre.ip_thetas_st[i]) * spectrum_pre.phis_num[i + 1];
		auto& cubes_father = octree.GetCubesLevel(i + 1);
		auto& cubes_son = octree.GetCubesLevel(i) ;
		int fa_index = 0;
		for (auto& cube_son : cubes_son) {
			while ((cube_son.mtc >> 3) != cubes_father[fa_index].mtc) ++fa_index;
			auto& cube_fa = cubes_father[fa_index];
			Vector3f r_phase = octree.mortoncode3d.GetGap(cube_son.mtc, i);
			int phase_shift_index = (cube_son.mtc & 7);
			if (phase_shift_now[phase_shift_index].size() > 0) continue;

			phase_shift_now[phase_shift_index].resize(spectrum_father_num);
			for (int l = 0; l < spectrum_father_num; ++l) {
				float kdotr = -(spectrum_pre.k_vecs[i + 1][l]).dot(r_phase);
				float ct = cos(kdotr), st = sin(kdotr);
				phase_shift_now[phase_shift_index][l] = CP(ct, st);
			}
		}
	}
	if (mpipre.GetRank() == 0) cout << "获得相位转移矩阵" << endl;
}

void Interpolation::GetTransfers()
{
	int level_num = spectrum_pre.GetTopLevelNum();
	transfers.reserve(level_num);
	float wavenumber = spectrum_pre.wavenumber;
	for (int i = 0; i < level_num; ++i)
	{
		auto& proxy_cubes = octree.GetProxyCubesLevel(i);
		auto& cubes = octree.GetCubesLevel(i);
		std::vector<Eigen::VectorXcf> transfers_level;
		std::vector<float> transfers_D;
		transfers_level.reserve(25);
		int L = GetL(spectrum_pre.length * sqrt(3.0) * (1 << level_num), wavenumber);
		int spectrum_num = spectrum_pre.k_vecs[i].size();
		auto& weights = spectrum_pre.weights[i];
		for (auto& proxy_cube : proxy_cubes) {
			for (auto& cube_index : proxy_cube.local_index) {
				Vector3f D_vec = octree.mortoncode3d.GetGap2(proxy_cube.mtc, cubes[cube_index].mtc, i);
				float D = D_vec.norm();
				int index = -1;
				for (int j = 0; j < transfers_D.size(); ++j) {
					if (abs(D - transfers_D[j]) < 1e-5) {
						index = j;
						break;
					}
				}
				if (index == -1) {
					transfers_D.push_back(D);
					proxy_cube.index.push_back(transfers_level.size() - 1);
					VectorXcf transfer(spectrum_num);
					float kD = wavenumber * D;
					if (L < 30) {
						for (int k = 0; k < spectrum_num; ++k)
						{
							float kdotD = (spectrum_pre.k_vecs[i][k] / wavenumber).dot(D_vec / D);
							CP TL(0, 0);
							if (kdotD > 1.0) kdotD -= 1e-8;
							if (kdotD < -1.0) kdotD += 1e-8;
							TL = TranferF(L, kdotD, kD, wavenumber) * weights[k];
							transfer[k] = TL;
						}
					}
					else {
						int LL = L > int(kD) ? int(kD) - 2 : L;
						int M = 5 * LL;
						float diff_theta = pi / M;
						vector<CP> TArray(M + 1);
						for (int k = 0; k <= M; ++k) {
							float ct = cos(diff_theta * k);
							TArray[k] = TranferF(LL, ct, kD, wavenumber);
						}
						for (int k = 0; k < spectrum_num; ++k)
						{
							float kdotD = (spectrum_pre.k_vecs[i][k] / wavenumber).dot(D_vec / D_vec.norm());
							CP TL(0, 0);
							if (kdotD > 1.0) kdotD -= 1e-8;
							if (kdotD < -1.0) kdotD += 1e-8;
							//TL = TranferF(L, kdotD, kD);
							TL = LagrangeITF(TArray, kdotD, diff_theta, 3) * weights[k];
							transfer[k] = TL;
						}
					}
					transfers_level.push_back(transfer);
				}
				else {
					proxy_cube.index.push_back(index);
				}
			}
		}
	}
}

void Interpolation::MemPre()
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
		
		if (i < mpipre.Get_HSP_end()) {
			size_t max_num = 0;
			int transfer_process_num = mpipre.GetTransferProcess(i).size();
			for (auto& local_cubes_sent_level : octree.local_cubes_sent[i]) max_num = max(max_num, local_cubes_sent_level.size());
			for (auto& local_cubes_recv_num : octree.far_cubes_recv_num[i]) max_num = max(max_num, size_t(local_cubes_recv_num));
			mpi_mem_size = max(mpi_mem_size, sptm_num * (cubes.size() + max_num * min(send_buffer_size + recv_buffer_size, transfer_process_num) * 2) * 2);
		}
		

		if (i < level_num - 1) {
			int itp_num = spectrum_pre.k_vecs[i].size();
			auto& cubes_fa = octree.GetCubesLevel(i + 1);
			mpi_mem_size = max(mpi_mem_size, itp_num * cubes_fa.size() * 2);
		}
	}
	mem_st_ptr = (CP*)malloc(mem_size * sizeof(CP));
	mem_end_ptr = mem_st_ptr + mem_size;
	mpi_mem_st_ptr = (CP*)malloc(mpi_mem_size * sizeof(CP));
	mpi_mem_end_ptr = mpi_mem_st_ptr + mpi_mem_size;
	cout << "分配内存成功 " << double(mem_size + mpi_mem_size) * 8.0 / 1024.0 / 1024.0 / 1024.0 << " GB" << endl;
}

void Interpolation::InterpolationProd(bool judge_hh, bool judge)
{
	int level_num = spectrum_pre.GetTopLevelNum();

	for (int i = 0; i < level_num - 1; i++) {
		int level = judge ? level_num - 2 - i : i;
		auto& cubes_fa = octree.GetCubesLevel(level + 1);
		auto& cubes_son = octree.GetCubesLevel(level);
		auto& proxy_cubes = octree.GetProxyCubesLevel(level);
		auto& local_cubes_sent = octree.local_cubes_sent[level];

		int Itp_theta_st = spectrum_pre.ip_thetas_st[level], Itp_theta_end = spectrum_pre.ip_thetas_end[level];
		int father_phi_num = spectrum_pre.phis_num[level + 1], son_phi_num = spectrum_pre.phis_num[level];
		int fa_theta_st = spectrum_pre.sptm_thetas_st[level + 1], fa_theta_end = spectrum_pre.sptm_thetas_end[level + 1];
		int fa_sptm_num = (fa_theta_end - fa_theta_st) * father_phi_num;

		int Itp_num = (Itp_theta_end - Itp_theta_st) * father_phi_num;
		int son_theta_st = spectrum_pre.sptm_thetas_st[level], son_theta_end = spectrum_pre.sptm_thetas_end[level];
		int sptm_son_num = (son_theta_end - son_theta_st) * son_phi_num;

		if (Itp_num != fa_sptm_num) {

		}

		if (judge) {

		}
		else {
			auto& Fu = Fu_list[level];
			auto& Fm = Fm_list[level];
			auto& Fd = Fd_list[level];
			auto& S = S_list[level];
			Eigen::VectorXcf temp_theta = VectorXcf::Zero(Fm.rows());
			Eigen::VectorXcf temp_phi = VectorXcf::Zero(Fm.rows());

			Map<VectorXcf> sptm_recv_up(temp_mem_ptr, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> sptm_recv_down(temp_mem_ptr + TWIP::p * 2 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> sptm_send_up(temp_mem_ptr + TWIP::p * 4 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> sptm_send_down(temp_mem_ptr + TWIP::p * 6 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> itp_sptm(temp_mem_ptr + TWIP::p * 8 * son_phi_num, Itp_num * 2);
			itp_sptm.setZero();
			MPI_Request request_s[4];
			MPI_Send_init(sptm_send_up.data(), 8 * son_phi_num, MPI_DOUBLE, mpipre.GetRankUp(level), 0, MPI_COMM_WORLD, &request_s[0]);
			MPI_Send_init(sptm_send_down.data(), 8 * son_phi_num, MPI_DOUBLE, mpipre.GetRankDown(level), 1, MPI_COMM_WORLD, &request_s[1]);
			MPI_Recv_init(sptm_recv_up.data(), 8 * son_phi_num, MPI_DOUBLE, mpipre.GetRankUp(level), 1, MPI_COMM_WORLD, &request_s[2]);
			MPI_Recv_init(sptm_recv_down.data(), 8 * son_phi_num, MPI_DOUBLE, mpipre.GetRankDown(level), 0, MPI_COMM_WORLD, &request_s[3]);

			int fa_index = 0;
			for (int j = 0; j < cubes_son.size(); j++) {
				//while((cubes_son[j].mtc >> 3) != cubes_fa[fa_index].mtc) ++fa_index;
				int phase_shift_index = (cubes_son[j].mtc & 7);
				VectorXcf& phase_shift = phase_shifts[level][phase_shift_index];

				Map<VectorXcf> sptm_son_theta(cubes_son[j].sptmtheta, sptm_son_num);
				Map<VectorXcf> sptm_son_phi(cubes_son[j].sptmphi, sptm_son_num);

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
				temp_theta.noalias() = Fu * sptm_son_theta;
				temp_phi.noalias() = Fu * sptm_son_phi;
				if (son_theta_st != 0) {
					MPI_Wait(&request_s[2], MPI_STATUS_IGNORE);
					temp_theta.noalias() += Fu * sptm_recv_up.topRows(TWIP::p * son_phi_num);
					temp_phi.noalias() += Fu * sptm_recv_up.bottomRows(TWIP::p * son_phi_num);
				}
				if (son_theta_end != spectrum_pre.thetas_num[level]) {
					MPI_Wait(&request_s[3], MPI_STATUS_IGNORE);
					temp_theta.noalias() += Fd * sptm_recv_down.topRows(TWIP::p * son_phi_num);
					temp_phi.noalias() += Fd * sptm_recv_down.bottomRows(TWIP::p * son_phi_num);
				}

				itp_sptm.topRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * temp_theta);
				itp_sptm.bottomRows(Itp_num).noalias() += phase_shift.cwiseProduct(S * temp_phi);

				if (son_theta_st != 0) MPI_Wait(&request_s[0], MPI_STATUS_IGNORE);
				if (son_theta_end != spectrum_pre.thetas_num[level]) MPI_Wait(&request_s[1], MPI_STATUS_IGNORE);

				while (j == cubes_son.size() - 1 || (cubes_son[j + 1].mtc >> 3) != cubes_fa[fa_index].mtc) {
					if (level == mpipre.Get_BP_end() - 1 && mpipre.GetProcessNum(level + 1) > 1) {
						//const int cluster_num = mpipre.GetProcessNum(level + 1);
						//std::vector<MPI_Request> request(cluster_num);
						//int theta_num = spectrum_pre.GetThetaNumLevel(level + 1);
						//double theta_dif = pi / (theta_num - 1);
						//CPD* s = spectrum_father.data();
						//Map<VectorXcd> spectrum_recv(spectrum_pre.temp_mem + 4 * p * 2 * father_phi_num + spectrum_father_num * 2, fa_s_num * 2);
						////if (mpipre.GetRank() == 0) cout << "? " <<cluster_num << endl;
						//for (int k = 0; k < cluster_num; k++) {
						//	double theta_st = pi / cluster_num * k, theta_end = pi / cluster_num * (k + 1);
						//	int spectrum_theta_st = k == 0 ? 0 : (theta_st + 1e-10) / theta_dif + 1;
						//	int spectrum_theta_end = (theta_end + 1e-10) / theta_dif + 1;
						//	int num = (spectrum_theta_end - spectrum_theta_st) * father_phi_num;

						//	if (mpipre.GetRank() != mpipre.Get_BP_com(k)) {
						//		MPI_Isend(s, num * 2 * 2, MPI_DOUBLE, mpipre.Get_BP_com(k), 0, MPI_COMM_WORLD, &request[k]);
						//	}
						//	else spectrum_fa.noalias() = spectrum_father.segment(spectrum_theta_st * father_phi_num * 2, num * 2);
						//	s += num * 2;
						//}

						//for (int k = 0; k < cluster_num; k++) {
						//	if (mpipre.GetRank() != mpipre.Get_BP_com(k)) {
						//		MPI_Recv(spectrum_recv.data(), fa_s_num * 2 * 2, MPI_DOUBLE, mpipre.Get_BP_com(k), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						//		MPI_Wait(&request[k], MPI_STATUS_IGNORE);
						//		spectrum_fa.noalias() += spectrum_recv;
						//	}
						//}

					}
					else {
						Map<VectorXcf> sptm_fa_theta(cubes_fa[j].sptmtheta, fa_sptm_num);
						Map<VectorXcf> sptm_fa_phi(cubes_fa[j].sptmphi, fa_sptm_num);

						if (Itp_num != fa_sptm_num) {
							int exchange_st = fa_theta_end != Itp_theta_end ? fa_theta_end : Itp_theta_st;
							int exchange_end = fa_theta_end != Itp_theta_end ? Itp_theta_end : fa_theta_st;
							int exchange_num = (exchange_end - exchange_st) * father_phi_num;
							int st = (exchange_st - Itp_theta_st) * father_phi_num;
							Map<VectorXcf> sptm_exchange(itp_sptm.data() + Itp_num * 2, exchange_num * 2);
							sptm_exchange.topRows(exchange_num) = itp_sptm.topRows(Itp_num).segment(st, exchange_num);
							sptm_exchange.bottomRows(exchange_num) = itp_sptm.bottomRows(Itp_num).segment(st, exchange_num);
							MPI_Sendrecv(sptm_exchange.data(), exchange_num * 2, MPI_COMPLEX, mpipre.GetRankExchange(level), j,
								cubes_fa[fa_index].sptmtheta, fa_sptm_num * 2, MPI_COMPLEX, mpipre.GetRankExchange(level), j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							sptm_fa_theta.noalias() += itp_sptm.topRows(Itp_num).segment((fa_theta_st - Itp_theta_st) * father_phi_num, fa_sptm_num);
							sptm_fa_phi.noalias() += itp_sptm.bottomRows(Itp_num).segment((fa_theta_st - Itp_theta_st) * father_phi_num, fa_sptm_num);
						}
						else {
							Map<VectorXcf> sptm_fa(cubes_fa[j].sptmtheta, fa_sptm_num * 2);
							sptm_fa.noalias() = itp_sptm;
						}
					}
					itp_sptm.setZero();
					++fa_index;
				}
			}
			//转移
			{
				std::vector<int>& transfer_process = mpipre.GetTransferProcess(level);
				int m0 = lower_bound(transfer_process.begin(), transfer_process.end(), mpipre.GetRank()) - transfer_process.begin();
				int process_num = transfer_process.size();
				int recv_index = 0;
				//将本地波谱转移给发送数组
				CP*const transfer_send_st = mpi_mem_st_ptr + sptm_son_num * 2 * cubes_son.size();
				size_t rest_mem = mpi_mem_end_ptr - transfer_send_st;
				size_t transfer_buffer = rest_mem / (send_buffer_size + recv_buffer_size);
				CP* const transfer_recv_st = transfer_send_st + transfer_buffer * send_buffer_size;
				std::queue<MPI_Request> request_send;
				std::queue<MPI_Request> request_recv;
				int recv_num = 0, send_num = 0;
				auto& send_cubes_index = octree.local_cubes_sent[i];
				auto& recv_cubes_num = octree.far_cubes_recv_num[i];
				auto& proxy_cubes = octree.GetProxyCubesLevel(i);
				Map<VectorXcf> cubes_temp(mpi_mem_st_ptr, sptm_son_num * 2 * cubes_son.size());
				Map<VectorXcf> cubes_sptm(cubes_son[0].sptmtheta, sptm_son_num * 2 * cubes_son.size());
				cubes_temp = cubes_sptm;
				//非阻塞发送和接收
				while (recv_num < process_num) {
					//将send_index对应盒子的波谱填到发送缓冲区中（用两个指针判断有没有填满），将缓冲区中的数据发送出去
					//进行非阻塞接收（用两个指针判断有没有接）
					//检查有没有接收到，如果接收到就进行转移,同时recv_index++,移动指针
					//检查有没有发送完
					CP* send_st = transfer_send_st;
					CP* recv_st = transfer_recv_st;
					while (send_num < process_num && request_send.size() < send_buffer_size) {
						int send_index = (send_num + m0 + 1) % process_num;
						int send_rank = transfer_process[send_index];
						auto& cubes_index = send_cubes_index[send_index];
						if (cubes_index.size() == 0) continue;
						for (int j = 0; j < cubes_index.size(); j++) {
							Map<VectorXcf> sptm_send(send_st + sptm_son_num * 2 * j, sptm_son_num * 2);
							Map<VectorXcf> sptm_cube(cubes_son[cubes_index[j]].sptmtheta, sptm_son_num * 2);
							sptm_send = sptm_cube;
						}
					    request_send.push(MPI_Request());
						MPI_Isend(send_st, sptm_son_num * 2 * cubes_index.size(), MPI_COMPLEX, send_rank, cubes_index.size(), MPI_COMM_WORLD, &request_send.back());
						send_st += transfer_buffer;
						++send_num;
					}
					while (request_recv.size() < recv_buffer_size && recv_num < process_num) {
						int recv_index = (m0 - recv_num - 1 + process_num) % process_num;
						if (recv_cubes_num[recv_index] == 0) continue;
						request_recv.push(MPI_Request());
						MPI_Irecv(recv_st, transfer_buffer, MPI_COMPLEX, transfer_process[recv_index], recv_cubes_num[recv_index], MPI_COMM_WORLD, &request_recv.back());
						recv_st += transfer_buffer;
						++recv_num;
					}
					recv_st = transfer_recv_st;
					MPI_Status status;
					while (!request_recv.empty() && MPI_Wait(&request_recv.front(), &status)) {
						int source_rank = status.MPI_SOURCE;
						int st = octree.far_cubes_process_index[i][source_rank];
						while (proxy_cubes[st].process_index == source_rank) {
							for (int j = 0; j < proxy_cubes[st].local_index.size(); j++) {
								Map<VectorXcf> sptm_recv(recv_st + sptm_son_num * 2 * j, sptm_son_num * 2);
								int local_cube_index = proxy_cubes[st].local_index[j];
								int transfer_index = proxy_cubes[st].index[j];
								Map<VectorXcf> sptm_cube(cubes_son[local_cube_index].sptmtheta, sptm_son_num * 2);
								sptm_cube.topRows(sptm_son_num).noalias() = sptm_recv.topRows(sptm_son_num).cwiseProduct(transfers[i][transfer_index]);
								sptm_cube.bottomRows(sptm_son_num).noalias() = sptm_recv.bottomRows(sptm_son_num).cwiseProduct(transfers[i][transfer_index]);
							}
							++st;
						}
						recv_st += transfer_buffer;
						request_recv.pop();
					}
					while (!request_send.empty() && MPI_Wait(&request_send.front(), MPI_STATUS_IGNORE)) request_send.pop();
				}
				
			}
			
			
		}
	}
}

