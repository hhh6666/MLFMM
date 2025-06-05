#include "MLFMM.h"
using namespace std;
using namespace Eigen;

void MLFMM::GmresP(const VecCP& b, VecCP& J)
{
	if (mpipre.GetRank() == 0) cout << "!!! " << b.squaredNorm() << endl;
	VecCP b_pre(b.rows());
	matrix_pre.SelfProd(b, b_pre.data());
	if (mpipre.GetRank() == 0) cout << "Start iterating " << b.squaredNorm() << endl;
	//b_pre = b;
	/*J = b_pre;
	cout << "电流范数" << b_pre.squaredNorm() << endl;
	return;*/
	//mpiout(b_pre.squaredNorm(), mpipre);
	//cout << b_pre.squaredNorm() << endl;
	clock_t start, end;
	start = clock();
	J.setZero();
	JD b_norm = 0, b_send = b_pre.squaredNorm();
	//cout << mpipre.GetRank() << "Zb_send" << Zb_send << endl;
	MPI_Reduce(&b_send, &b_norm, 1, MPI_JD, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&b_norm, 1, MPI_JD, 0, MPI_COMM_WORLD);
	//cout << mpipre.GetRank() << "Zb_norm" << Zb_norm << endl;
	b_norm = sqrt(b_norm);
	size_t restart_num = 5;
	size_t iterations_max = 100;
	size_t lieshu = J.rows();
	JD threshold = 1e-3;
	VecCP Hk = VecCP::Zero(iterations_max);
	for (int j = 0; j < restart_num; ++j) {
		VecCP r = b_pre;
		if (j > 0) {
			Prod(J.data(), b_temp.data());
			r -= b_temp;
		}
		VecCP c = VecCP::Zero(iterations_max);
		VecCP s = VecCP::Zero(iterations_max);
		VecCP e1 = VecCP::Zero(iterations_max);
		VecCP beta = VecCP::Zero(iterations_max);
		MatCP H = MatCP::Zero(iterations_max, iterations_max);
		MatCP V = MatCP::Zero(lieshu, iterations_max);
		e1[0] = CP(1.0, 0);
		JD r_norm = 0, r_send = r.squaredNorm();
		MPI_Reduce(&r_send, &r_norm, 1, MPI_JD, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(&r_norm, 1, MPI_JD, 0, MPI_COMM_WORLD);
		r_norm = sqrt(r_norm);
		JD error = r_norm / b_norm;
		int number = iterations_max - 2;
		//std::cout << error << " " << r_norm << std::endl;
		V.col(0) = r / r_norm;
		beta = r_norm * e1;
		for (int k = 0; k < iterations_max - 1; ++k) {
			Prod(V.col(k).data(), V.col(k + 1).data());
			for (int i = 0; i <= k; ++i) {
				Hk(i) = V.col(i).dot(V.col(k + 1));
			}
			MPI_Reduce(Hk.data(), H.col(k).data(), (k + 1) * 2, MPI_JD, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Bcast(H.col(k).data(), (k + 1) * 2, MPI_JD, 0, MPI_COMM_WORLD);
			//cout << mpipre.GetRank() << "???" << endl;
			V.col(k + 1) -= V.leftCols(k + 1) * H.col(k).segment(0, k + 1);

			JD local_send = V.col(k + 1).squaredNorm(), local_recv = 0.0;
			MPI_Reduce(&local_send, &local_recv, 1, MPI_JD, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Bcast(&local_recv, 1, MPI_JD, 0, MPI_COMM_WORLD);
			//cout << mpipre.GetRank() << "????" << endl;
			H(k + 1, k) = sqrt(local_recv);
			if (abs(H(k + 1, k)) < 1e-10) {
				number = k;
				break;
			}
			V.col(k + 1) /= H(k + 1, k);
			for (size_t i = 0; i < k; ++i) {
				CP temp = c[i] * H(i, k) + s[i] * H(i + 1, k);
				//H(i + 1, k) = -s[i] * H(i, k) + c[i] * H(i + 1, k);
				H(i + 1, k) = -conj(s[i]) * H(i, k) + conj(c[i]) * H(i + 1, k);
				H(i, k) = temp;
			}
			JD tao = sqrt(norm(H(k, k)) + norm(H(k + 1, k)));

			/*c[k] = H(k, k) / tao;
			s[k] = H(k + 1, k) / tao;*/
			c[k] = abs(H(k, k)) * exp(cpdj * arg(H(k + 1, k))) / tao;
			s[k] = abs(H(k + 1, k)) * exp(cpdj * arg(H(k, k))) / tao;

			H(k, k) = c[k] * H(k, k) + s[k] * H(k + 1, k);
			H(k + 1, k) = CP(0, 0);

			/*beta[k + 1] = -s[k] * beta[k];
			beta[k] = c[k] * beta[k];*/
			beta[k + 1] = -conj(s[k]) * beta[k];
			beta[k] = c[k] * beta[k];

			error = abs(beta[k + 1]) / b_norm;
			if (mpipre.GetRank() == 0) std::cout << k << "误差 " << error << std::endl;
			end = clock();
			if (mpipre.GetRank() == 0) std::cout << "time = " << JD(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
			if (error < threshold) {
				number = k;
				break;
			}
		}

		++number;
		MatCP h = H.block(0, 0, number, number);
		//VecCP y = h.inverse() * beta.topRows(number);
		Eigen::PartialPivLU<MatCP> lu(h);
		VecCP y = lu.solve(beta.topRows(number));
		J += V.leftCols(number) * y;
		//Prod(J, b_temp.data());
		//cout << "最终误差 " << (b_temp - b).norm() / b.norm() << endl;
		if (error < threshold) {
			break;
		}
	}
}

