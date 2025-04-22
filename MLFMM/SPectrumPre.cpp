#include "SpectrumPre.h"

using namespace Eigen;
using namespace std;

void SpectrumPre::Get_Spectrum()
{
	cout<< mpipre.GetRank() <<"Get_Spectrum"<<endl;
	thetas.reserve(level_num + 1);
	k_vecs.reserve(level_num);
	weights.reserve(level_num);
	thetas_num.reserve(level_num);
	phis_num.reserve(level_num);
	sptm_thetas_st.reserve(level_num);
	sptm_thetas_end.reserve(level_num);
	ip_thetas_st.reserve(level_num - 1);
	ip_thetas_end.reserve(level_num - 1);
	for (int i = 0; i < level_num; ++i) {
		int L = GetL(length * JD(1 << i) * sqrt(3.0), GlobalParams.k0);
		int theta_num = GetThetaNum(L);//1.5, 
		int phi_num = GetPhiNum(L);

		vector<JD> thetas_level(theta_num);
		LegendrePolynomial LP(theta_num);
		for (int i = 0; i < theta_num; i++) {
			thetas_level[i] = acos(-LP.root(theta_num - i));
		}
		thetas.push_back(thetas_level);

		std::vector<VecJD3> k_unit;
		std::vector<JD> weights_level;
		JD theta_dif = pi / (theta_num - 1);
		JD phi_dif = 2 * pi / phi_num;
		if (mpipre.GetRank() == 0) cout << "theta数量" << theta_num << endl;
		int process_num = mpipre.GetProcessNum(i);
		int index = mpipre.GetProcessIndex(i);
		JD theta_st = pi / process_num * index, theta_end = pi / process_num * (index + 1);

		int spectrum_theta_st = upper_bound(thetas_level.begin(), thetas_level.end(), theta_st) - thetas_level.begin();
		int spectrum_theta_end = upper_bound(thetas_level.begin(), thetas_level.end(), theta_end) - thetas_level.begin();
		weights_level.reserve((spectrum_theta_end - spectrum_theta_st) * phi_num);
		for (int j = spectrum_theta_st; j < spectrum_theta_end; j++) {
			for (int k = 0; k < phi_num; k++) {
				weights_level.push_back(LP.weight(theta_num - j) * phi_dif);
			}
		}
		weights.push_back(weights_level);
		thetas_num.push_back(theta_num);
		phis_num.push_back(phi_num);
		sptm_thetas_st.push_back(spectrum_theta_st);
		sptm_thetas_end.push_back(spectrum_theta_end);
		if (i > 0) {
			int son_process_num = mpipre.GetProcessNum(i - 1);
			int son_index = mpipre.GetProcessIndex(i - 1);
			JD theta_st_son = pi / son_process_num * son_index, theta_end_son = pi / son_process_num * (son_index + 1);
			int interpolation_theta_st = upper_bound(thetas_level.begin(), thetas_level.end(), theta_st_son) - thetas_level.begin();
			int interpolation_theta_end = upper_bound(thetas_level.begin(), thetas_level.end(), theta_end_son) - thetas_level.begin();

			ip_thetas_st.push_back(interpolation_theta_st);
			ip_thetas_end.push_back(interpolation_theta_end);
			
			k_unit.reserve((interpolation_theta_end - interpolation_theta_st) * phi_num);
			for (int j = interpolation_theta_st; j < interpolation_theta_end; j++) {
				for (int k = 0; k < phi_num; k++) {
					k_unit.push_back(GetK(thetas_level[j], k * phi_dif, GlobalParams.k0));
				}
			}
		}
		else {
			k_unit.reserve(theta_num * phi_num);
			for (int j = 0; j < theta_num; j++) {
				for (int k = 0; k < phi_num; k++) {
					k_unit.push_back(GetK(thetas_level[j], k * phi_dif, GlobalParams.k0));
				}
			}
		}
		k_vecs.push_back(k_unit);
		//cout << i << "数量 " << k_unit.size() << endl;
	}

	int L = GetLH(length * sqrt(3.0), GlobalParams.k0);
	int theta_num = GetThetaNum(L);
	int phi_num = GetPhiNum(L);
	if (mpipre.GetRank() == 0) cout << "theta数量" << theta_num << endl;

	vector<JD> thetas_level(theta_num);
	LegendrePolynomial LP(theta_num);
	for (int i = 0; i < theta_num; i++) {
		thetas_level[i] = acos(-LP.root(theta_num - i));
	}
	thetas.push_back(thetas_level);
	thetas_num.push_back(theta_num);
	phis_num.push_back(phi_num);
	std::vector<VecJD3> k_need;
	k_need.reserve(theta_num * phi_num);
	TCtoS.reserve(theta_num * phi_num);
	JD phi_dif = 2 * pi / phi_num;
	for (int j = 0; j < theta_num; j++) {
		for (int k = 0; k < phi_num; k++) {
			JD ct = cos(thetas_level[j]);
			JD st = sin(thetas_level[j]);
			JD cp = cos(k * phi_dif);
			JD sp = sin(k * phi_dif);
			k_need.push_back(GlobalParams.k0 * VecJD3{ st * cp, st * sp, ct });
			Matrix<JD, 2, 3> T;
			T << ct * cp, ct* sp, -st,
				-sp, cp, 0;
			TCtoS.push_back(T);
		}
	}
	k_vecs.push_back(k_need);

	if (mpipre.GetRank() == 0) std::cout << "波谱预处理完成" << endl;
}


void TWIP::GetMatrix(int fa_theta_st, int fa_theta_end, int son_theta_st, int son_theta_end)
{
	JD fa_phi_dif = 2 * pi / fa_phi_num;
	JD son_phi_dif = 2 * pi / son_phi_num;
	
	int u_theta_st = max(fa_theta_st - pp, 0), u_theta_end = min(fa_theta_end + pp, int(fa_thetas.size()));
	int d_theta_st = max(son_theta_st - p, 0), d_theta_end = min(son_theta_end + p, int(son_thetas.size()));

	int spectrum_father_num = (u_theta_end - u_theta_st) * fa_phi_num;
	int spectrum_temp_num = (u_theta_end - u_theta_st) * son_phi_num;
	int spectrum_son_num = (d_theta_end - d_theta_st) * son_phi_num;

	First.resize(spectrum_temp_num, spectrum_son_num);
	Second.resize(spectrum_father_num, spectrum_temp_num);
	vector<T_> CList_F, CList_S;
	CList_F.reserve(spectrum_temp_num * 4);
	CList_S.reserve(spectrum_father_num * 4);

	int son_shift = d_theta_st * son_phi_num;

	for (int i = u_theta_st; i < u_theta_end; i++) {
		JD theta = fa_thetas[i];
		int m0 = std::lower_bound(son_thetas.begin(), son_thetas.end(), theta + 1e-7) - son_thetas.begin() - 1;//第一个大于等于theta的元素的下标
		for (int j = 0; j < son_phi_num; j++) {
			int fa_index = (i - u_theta_st) * son_phi_num + j;
			int mbegin = son_theta_st ? max(m0 - p + 1, d_theta_st) : m0 - p + 1, mend = son_theta_end < int(son_thetas.size()) ? min(m0 + p, d_theta_end - 1) : m0 + p;
			//int mbegin = m0 - p + 1, mend = m0 + p;
			for (int m = mbegin; m <= mend; m++) {
				JD mtheta = GetTheta(m);
				int son_index = GetIndex(m, j) - son_shift;
				JD coef = IsVirtual(m);
				for (int a = m0 - p + 1; a <= m0 + p; a++) {
					if (a == m) continue;
					JD atheta = GetTheta(a);
					coef *= (theta - atheta) / (mtheta - atheta);
				}
				CList_F.push_back(T_{ fa_index, son_index, coef });
			}
		}
	}
	for (int i = u_theta_st; i < u_theta_end; i++) {
		for (int j = 0; j < fa_phi_num; j++) {
			int fa_index = (i - u_theta_st) * fa_phi_num + j;
			JD phi = j * fa_phi_dif;
			int n0 = phi / son_phi_dif;
			for (int n = n0 - p + 1; n <= n0 + p; n++) {
				JD nphi = n * son_phi_dif;
				int son_index = (i - u_theta_st) * son_phi_num + (n + son_phi_num) % son_phi_num;
				JD coef = 1.0;
				for (int b = n0 - p + 1; b <= n0 + p; b++) {
					if (b == n) continue;
					JD bphi = b * son_phi_dif;
					coef *= (phi - bphi) / (nphi - bphi);
				}
				CList_S.push_back(T_{ fa_index, son_index, coef });
			}
		}
	}
	First.setFromTriplets(CList_F.begin(), CList_F.end());
	Second.setFromTriplets(CList_S.begin(), CList_S.end());

	/*int son_shift = son_theta_st ? p * son_phi_num : 0;
	int fa_shift = fa_theta_st ? pp * fa_phi_num : 0;
	int actual_fa_spectrum_num = (fa_theta_end - fa_theta_st) * fa_phi_num;
	int actual_temp_spectrum_num = (fa_theta_end - fa_theta_st) * son_phi_num;
	int actual_son_spectrum_num = (son_theta_end - son_theta_st) * son_phi_num;
	Fu = First.middleRows(fa_shift, actual_temp_spectrum_num).leftCols(p * son_phi_num);
	Fm = First.middleRows(fa_shift, actual_temp_spectrum_num).middleCols(son_shift, actual_son_spectrum_num);
	Fd = First.middleRows(fa_shift, actual_temp_spectrum_num).rightCols(p * son_phi_num);
	S = Second.block(fa_shift, fa_shift, actual_fa_spectrum_num, actual_temp_spectrum_num);

	aFu = First.transpose().middleRows(son_shift, actual_son_spectrum_num).leftCols(pp * fa_phi_num);
	aFm = First.transpose().middleRows(son_shift, actual_son_spectrum_num).middleCols(fa_shift, actual_fa_spectrum_num);
	aFd = First.transpose().middleRows(son_shift, actual_son_spectrum_num).rightCols(pp * fa_phi_num);
	aS = Second.transpose();*/
}