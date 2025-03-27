#include "MatrixPre.h"

using namespace std;
using namespace Eigen;

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
	transfers_index.reserve(level_num);
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
					proxy_cube.transfers_index.push_back(transfers_level.size() - 1);
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
					proxy_cube.transfers_index.push_back(index);
				}
			}
		}
	}
}

void Interpolation::InterpolationProd(bool judge_hh, bool judge)
{
	int level_num = spectrum_pre.GetTopLevelNum();

	for (int i = 0; i < level_num; i++) {
		int level = judge ? level_num - 2 - i : i;
		auto& cubes_fa = octree.GetCubesLevel(level + 1);
		auto& cubes_son = octree.GetCubesLevel(level);
		auto& proxy_cubes = octree.GetProxyCubesLevel(level);
		auto& local_cubes_sent = octree.GetLocalCubesSent(level);

		int Itp_theta_st = spectrum_pre.ip_thetas_st[level], Itp_theta_end = spectrum_pre.ip_thetas_end[level];
		int father_phi_num = spectrum_pre.phis_num[level + 1], son_phi_num = spectrum_pre.phis_num[level];
		int fa_theta_st = spectrum_pre.sptm_thetas_st[level + 1], fa_theta_end = spectrum_pre.sptm_thetas_end[level + 1];
		int fa_sptm_num = (fa_theta_end - fa_theta_st) * father_phi_num;

		int Itp_num = (Itp_theta_end - Itp_theta_st) * father_phi_num;
		int son_theta_st = spectrum_pre.sptm_thetas_st[level], son_theta_end = spectrum_pre.sptm_thetas_end[level];
		int spectrum_son_num = (son_theta_end - son_theta_st) * son_phi_num;

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

			Map<VectorXcf> sptm_recv_up(spectrum_pre.temp_mem, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> sptm_recv_down(spectrum_pre.temp_mem + TWIP::p * 2 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> sptm_send_up(spectrum_pre.temp_mem + TWIP::p * 4 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> sptm_send_down(spectrum_pre.temp_mem + TWIP::p * 6 * son_phi_num, TWIP::p * 2 * son_phi_num);
			Map<VectorXcf> itp_sptm(spectrum_pre.temp_mem + TWIP::p * 8 * son_phi_num, Itp_num * 2);
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

				Map<VectorXcf> sptm_son_theta(cubes_son[j].sptmtheta, spectrum_son_num);
				Map<VectorXcf> sptm_son_phi(cubes_son[j].sptmphi, spectrum_son_num);

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
			if (level < mpipre.Get_HSP_end()) {
				auto& transfer_process = mpipre.GetTransferProcess(level);
				int m0 = lower_bound(transfer_process.begin(), transfer_process.end(), mpipre.GetRank()) - transfer_process.begin();
				int process_num = transfer_process.size();
				for (int j = 0; j < process_num; j++) {
					int send_rank = transfer_process[(j + m0 + 1) % process_num];
					int recv_rank = transfer_process[(j + m0 - 1 + process_num) % process_num];
					//预处理两个数组，一个负责发送，一个负责接收
				}
			}
			
		}
	}
}