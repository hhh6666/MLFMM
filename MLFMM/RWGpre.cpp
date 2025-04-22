#include "RWGpre.h"

using namespace std;
using namespace Eigen;

RWG& RWG::ReadNas(std::string filename, int triangle_num)
{
	std::ifstream fin(filename, std::ios::in);
	if (!fin.is_open()) {
		std::cout << "无法找到文件:" << filename << std::endl;
		return *this;
	}
	std::string buff;

	points.reserve((triangle_num + 4) / 2);
	triangles.reserve(triangle_num);
	edges.reserve(triangle_num * 1.5);

	VecJD3 point;
	Vector3i vertex;
	while (getline(fin, buff))
	{
		if (buff[0] == 'G')
		{
			point(0) = std::stod(buff.substr(40, 16));
			point(1) = std::stod(buff.substr(56, 16));
		}
		else if (buff[0] == '*')
		{
			point(2) = std::stod(buff.substr(8, 16));
			start_point[0] = min(start_point[0], point(0)), start_point[1] = min(start_point[1], point(1));
			start_point[2] = min(start_point[2], point(2)), end_point[0] = max(end_point[0], point(0));
			end_point[1] = max(end_point[1], point(1)), end_point[2] = max(end_point[2], point(2));
			points.push_back(point);
		}
		else if (buff[0] == 'C')
		{
			std::istringstream line(buff);
			std::string word;
			for (int i = 1; i <= 6; ++i) {
				if (i == 4) {
					line >> vertex[0];
				}
				else if (i == 5) {
					line >> vertex[1];
				}
				else if (i == 6) {
					line >> vertex[2];
				}
				else {
					line >> word;
				}
			}
			triangles.push_back(vertex - Vector3i(1, 1, 1));
		}
	}
	fin.close();
	std::unordered_map<int, std::unordered_map<int, int>> edge_map{};
	for (int k = 0; k < triangles.size(); ++k) {
		Vector3i& triangle = triangles[k];
		for (int i = 0; i < 3; ++i) {
			for (int j = i + 1; j < 3; ++j) {
				int v1 = triangle[i], v2 = triangle[j];
				if (v1 > v2) swap(v1, v2);
				if (edge_map[v1][v2] == 0) {
					Vector2i edge(k, -1);
					edges.push_back(edge);
					edge_map[v1][v2] = edges.size();
					//edges_length.push_back((points[v1] - points[v2]).norm());
				}
				else if (edge_map[v1][v2] > 0) {
					edges[edge_map[v1][v2] - 1][1] = k;
					edge_map[v1][v2] = -1;
				}
			}
		}
	}
	//cout << "edges.size() =" << edges.size() << " " << triangle_num * 1.5 << endl;
	edges.erase(std::remove_if(edges.begin(), edges.end(), [](const Vector2i& x) {return x[1] == -1; }), edges.end());
	
	//cout << "triangles.size()=" << triangles.size() << " " << triangle_num << endl;
	//cout << "points.size()=" << points.size() << " " << (triangle_num + 4) / 2 << endl;
	//cout << "edges.size()=" << edges.size() << " " << triangle_num * 1.5 << endl;
	points.shrink_to_fit();
	triangles.shrink_to_fit();
	edges.shrink_to_fit();
	cout << "读取文件成功" << endl;
	//cout << start_point.transpose() << "       " << end_point.transpose() << endl;
	return *this;
}

JD RWG::TriDistence(int i, int j)
{
	if (i == j) return 0.0;
	VecJD3 center1 = (points[triangles[i][0]] + points[triangles[i][1]] + points[triangles[i][2]]) / 3.0;
	VecJD3 center2 = (points[triangles[j][0]] + points[triangles[j][1]] + points[triangles[j][2]]) / 3.0;
	return (center1 - center2).norm();
}


const VecJD3 RWG::GetEdgeCenter(int edge_index) const
{
	int tri1 = edges[edge_index][0], tri2 = edges[edge_index][1];
	VecJD3 center1 = (points[triangles[tri1][0]] + points[triangles[tri1][1]] + points[triangles[tri1][2]]) / 3.0;
	VecJD3 center2 = (points[triangles[tri2][0]] + points[triangles[tri2][1]] + points[triangles[tri2][2]]) / 3.0;
	return (center1 + center2) / 2;
}

RWG& RWG::initial()
{
	Matrix<JD, 3, 3> l;
	Matrix<JD, 3, 3> u;
	tri_l.reserve(triangles.size());
	tri_l_normal.reserve(triangles.size());
	tri_area.reserve(triangles.size());
	tri_normal.reserve(triangles.size());
	for (int i = 0; i < triangles.size(); ++i) {
		VecJD3& v1 = points[triangles[i][0]];
		VecJD3& v2 = points[triangles[i][1]];
		VecJD3& v3 = points[triangles[i][2]];
		l.col(0) = (v2 - v1).normalized();
		l.col(1) = (v3 - v2).normalized();
		l.col(2) = (v1 - v3).normalized();
		VecJD3 n = (v2 - v1).cross((v3 - v2));
		u.col(0) = l.col(0).cross(n).normalized();
		u.col(1) = l.col(1).cross(n).normalized();
		u.col(2) = l.col(2).cross(n).normalized();
		tri_l.push_back(l);
		tri_l_normal.push_back(u);
		tri_area.push_back(n.norm() / 2.0);
		tri_normal.push_back(n.normalized());
	}

	Matrix<JD, 3, 6> g;
	Vector2i edge_vertex;
	J_GL.reserve(edges.size());
	vertex_edges.reserve(edges.size());
	for (int i = 0; i < edges.size(); ++i) {
		for (int j = 0; j < 2; ++j) {
			Vector3i& tri = triangles[edges[i][j]];
			for (int k = 0; k < 3; ++k) {
				g.col(j * 3 + k) = GL_points[k][1] * points[tri[0]] + GL_points[k][2] * points[tri[1]]
					+ GL_points[k][3] * points[tri[2]];
			}
		}
		Vector3i& tri1 = triangles[edges[i][0]], tri2 = triangles[edges[i][1]];
		vector<int> count(6, 0);
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				if (tri1[j] == tri2[k]) {
					count[j] = count[k + 3] = 1;
					break;
				}
			}
		}
		for (int j = 0; j < 3; ++j) if (count[j] == 0) {
			edge_vertex[0] = tri1[j];
			int a = (j + 1) % 3, b = (j + 2) % 3;
			edges_length.push_back((points[tri1[a]] - points[tri1[b]]).norm());
		}
		for (int j = 0; j < 3; ++j) if (count[j + 3] == 0) edge_vertex[1] = tri2[j];
		J_GL.push_back(g);
		vertex_edges.push_back(edge_vertex);
	}
	//cout << "size" << J_GL.size() << " " << vertex_edges.size() << endl;
	//cout << "初始化完成" << endl;
	return *this;
}

Eigen::Vector<JD, 4> RWG::GetI1I2(const VecJD3& r, int tri_index)
{
	Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
	for (int i = 0; i < 3; ++i) {
		VecJD3& v1 = points[triangles[tri_index][i]];
		VecJD3& v2 = points[triangles[tri_index][(i + 1) % 3]];
		JD lp = (v2 - r).dot(tri_l[tri_index].col(i));
		JD lm = (v1 - r).dot(tri_l[tri_index].col(i));
		JD Rp = (v2 - r).norm();
		JD Rm = (v1 - r).norm();
		JD d = abs(tri_normal[tri_index].dot(r - v2));
		JD P0 = abs((v1 - r).dot(tri_l_normal[tri_index].col(i)));
		JD R0 = sqrt(d * d + P0 * P0);
		JD f = log((Rp + lp) / (Rm + lm));
		JD beta = atan(P0 * lp / (R0 * R0 + d * Rp)) - atan(P0 * lm / (R0 * R0 + d * Rm));
		ans.head(3) += tri_l_normal[tri_index].col(i) * (R0 * R0 * f + lp * Rp - lm * Rm);
		JD P0u = (v1 - r).dot(tri_l_normal[tri_index].col(i)) / P0;
		ans[3] += P0u * (P0 * f - d * beta);
	}
	ans.head(3) *= 0.5;
	return ans;
}

CP RWG::GetZij(int i, int j)
{
	JD lam = GlobalParams.lam;
	JD k0 = GlobalParams.k0;
	CP C1 = Zf * k0 * 0.25 * cpdj;
	CP C2 = Zf / k0 * cpdj;
	Eigen::Matrix<JD, 3, 6>& Ji = J_GL[i];
	Eigen::Matrix<JD, 3, 6>& Jj = J_GL[j];
	CP Zij = CP(0.0, 0.0);
	for (int m = 0; m < 2; ++m) {
		JD pmm = m ? -1.0 : 1.0;
		for (int n = 0; n < 2; ++n) {
			JD pmn = n ? -1.0 : 1.0;
			JD Rij = TriDistence(edges[i][m], edges[j][n]);
			JD pm = m == n ? 1.0 : -1.0;
			for (int p = 0; p < GLPN; ++p) {
				//if (Rij < lam * 0.03) cout << "Zij=" << Zij << " "<<m<<" "<<n<<endl;
				if (Rij < eps*lam) {
					//cout << i << " " << j << endl;
					VecJD3 Jm = pmm * (Ji.col(m * 3 + p) - points[vertex_edges[i][m]]);
					Vector<JD, 4> I1I2 = GetI1I2(Ji.col(m * 3 + p), edges[j][n]);
					VecCP3 a = C1 * Jm * pmn;
					CP b = -pm * C2 + pmn * C1 * Jm.dot(Ji.col(m * 3 + p) - points[vertex_edges[j][n]]);
					Zij += GL_points[p][0] * (I1I2.head(3).dot(a) + b * I1I2[3]) / tri_area[edges[j][n]];
					//cout << "Zij=" << Zij << " "<< I1I2.transpose()<< endl;
				}
				//cout << i << "  " << j << endl;
				for (int q = 0; q < GLPN; ++q) {
					VecJD3 Jm = pmm * (Ji.col(m * 3 + p) - points[vertex_edges[i][m]]);
					VecJD3 Jn = pmn * (Jj.col(n * 3 + q) - points[vertex_edges[j][n]]);
					JD Rpq = (Ji.col(m * 3 + p) - Jj.col(n * 3 + q)).norm();
					JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
					CP G = Rij < lam * 0.03 ? CP(-0.5 * k0 * k0 * Rpq, k0 * k0 * k0 * Rpq * Rpq / 6.0 - k0)
						: CP(ct, st) / Rpq;
					Zij += GL_points[p][0] * GL_points[q][0] * (C1 * Jm.dot(Jn) - pm * C2) * G;
					//if (i == j) cout << Zij << "  " << m << "  " << n << " " << G << " "<< Rpq<<endl;
				}

			}
		}
	}
	Zij *= edges_length[i] * edges_length[j] / (4.0 * pi);
	return Zij;
}

void MOM::Fillb(double theta, double phi)
{
	double ct = cos(theta * rad), st = sin(theta * rad), cp = cos(phi * rad), sp = sin(phi * rad);
	Vector3d r = Vector3d(st * cp, st * sp, ct);//方向相反
	b.setZero(rwg.edges.size());
	MatrixXd T(2, 3);
	T << ct * cp, ct* sp, -st,
		-sp, cp, 0;
	//T.transposeInPlace();
	Vector2d polor{ -1.0,0.0 };
	Vector3d plane_wave = T.transpose() * polor;
	for (int i = 0; i < rwg.edges.size(); i++) {
		Eigen::Matrix<JD, 3, 6>& Ji = rwg.J_GL[i];
		for (int m = 0; m < 2; ++m) {
			double pmm = m ? -1.0 : 1.0;
			for (int p = 0; p < GLPN; ++p) {
				Vector3d Jm = pmm * (Ji.col(m * 3 + p) - rwg.points[rwg.vertex_edges[i][m]]);
				JD rpr = GlobalParams.k0 * Ji.col(m * 3 + p).dot(r);
				CP ejkr = CP(cos(rpr), sin(rpr));
				b[i] += GL_points[p][0] * Jm.dot(plane_wave) * ejkr;
			}
		}
		b[i] *= rwg.edges_length[i] * 0.5;
	}
}

Eigen::Vector2cd MOM::FarField(double theta, double phi)
{
	double ct = cos(theta * rad), st = sin(theta * rad), cp = cos(phi * rad), sp = sin(phi * rad);
	Vector3d r = Vector3d(st * cp, st * sp, ct);
	Vector3cd e = Vector3cd::Zero();
	MatrixXd T(2, 3);
	T << ct * cp, ct* sp, -st,
		-sp, cp, 0;
	for (int i = 0; i < rwg.edges.size(); i++) {
		Eigen::Matrix<double, 3, 6>& Ji = rwg.J_GL[i];
		for (int m = 0; m < 2; ++m) {
			double pmm = m ? -1 : 1;
			for (int p = 0; p < GLPN; ++p) {
				Vector3d Jm = pmm * (Ji.col(m * 3 + p) - rwg.points[rwg.vertex_edges[i][m]]);
				double inrpr = GlobalParams.k0 * Ji.col(m * 3 + p).dot(r);
				double ct = cos(inrpr), st = sin(inrpr);
				e += GL_points[p][0] * rwg.edges_length[i] * x(i) * Jm * CP(ct, st) * 0.5;
			}
		}
	}
	return T * (-cpdj * Zf * GlobalParams.k0) / (4.0 * pi) * e;
}

void MOM::FillZ()
{
	int size = rwg.edges.size();
	Z.resize(size, size);
	for (int k = 0; k < size; k++) {
		for (int j = k; j < size; j++) {
			Z(j, k) = rwg.GetZij(j, k);
		}
	}
	for (int k = 0; k < size; k++) {
		for (int j = k; j < size; j++) {
			Z(k, j) = Z(j, k);
		}
	}

	Z_index.resize(N+1);
	lu_list.resize(N);
	for (int i = 0; i < N; ++i) {
		Z_index[i] = i * size / N;
		cout << Z_index[i] << endl;
	}
	Z_index[N] = size;
	for (int i = 0; i < N; ++i) {
		int block_size = Z_index[i + 1] - Z_index[i];
		Eigen::PartialPivLU<MatCP> Zinv(Z.block(Z_index[i], Z_index[i], block_size, block_size));
		lu_list[i] = Zinv;
		//if(i==0)cout << Zinv.solve(Z.block(Z_index[i], Z_index[i], block_size, block_size)).block(0,0,5,5) << endl;
	}
}

void MOM::SelfProd(VecCP& b, CP* b_str)
{
	Map<VecCP> b_map(b_str, b.size());
	//b_map = b;
	for (int i = 0; i < N; i++) {
		int block_size = Z_index[i + 1] - Z_index[i];
		b_map.segment(Z_index[i], block_size) = lu_list[i].solve(b.segment(Z_index[i], block_size));
	}
}

void MOM::Prod(const VecCP& x, CP* b_str)
{
	Map<VecCP> b_map(b_str, x.size());
	b_temp.setZero();
	//for (int i = 0; i < 8; i++) {
	//	int block_size_i = Z_index[i + 1] - Z_index[i];
	//	//b_temp.segment(Z_index[i], block_size_i).noalias() += Z.block(Z_index[i], Z_index[i], block_size_i, block_size_i) * x.segment(Z_index[i], block_size_i);
	//	
	//	for (int j = 0; j < 8; ++j) {
	//		int block_size_j = Z_index[j + 1] - Z_index[j];
	//		if (i != j) {
	//			b_temp.segment(Z_index[i], block_size_i).noalias() += Z.block(Z_index[i], Z_index[j], block_size_i, block_size_j) * x.segment(Z_index[j], block_size_j);
	//			//b_map.segment(Z_index[i], block_size_i).noalias() += lu_list[i].solve(x.segment(Z_index[j], block_size_j));
	//		}
	//	}
	//	
	//}
	b_temp = Z * x;
	SelfProd(b_temp, b_str);
	//b_map.noalias() += x;
}

void MOM::Gmres(VecCP& J, VecCP& b)
{
	cout << "Start iterating" << endl;
	
	Eigen::VectorXcd Zb(b.size());
	//Zb = Z.adjoint() * b;
	SelfProd(b, Zb.data());
	cout << "Zb done" << " " << Zb.squaredNorm() << endl;

	//Eigen::MatrixXcd ZZ = Z.adjoint() * Z;
	clock_t start, end;
	start = clock();
	J.setZero();
	JD Zb_norm = Zb.norm();
	size_t restart_num = 1;
	size_t iterations_max = 100;
	size_t lieshu = J.rows();
	JD threshold = 1e-3;
	Eigen::VectorXcd Hk = Eigen::VectorXcd::Zero(iterations_max);
	for (int j = 0; j < restart_num; ++j) {
		Eigen::VectorXcd r = Zb;
		//if (j > 0) r -= ConjProdProd(J);
		Eigen::VectorXcd c = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::VectorXcd s = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::VectorXcd e1 = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::VectorXcd beta = Eigen::VectorXcd::Zero(iterations_max);
		Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(iterations_max, iterations_max);
		Eigen::MatrixXcd V = Eigen::MatrixXcd::Zero(lieshu, iterations_max);
		e1[0] = CP(1.0, 0);
		JD r_norm = r.norm();
		JD error = r_norm / Zb_norm;
		int number = iterations_max - 2;
		//std::cout << error << " " << r_norm << std::endl;
		V.col(0) = r / r_norm;
		beta = r_norm * e1;
		for (int k = 0; k < iterations_max - 1; ++k) {
			//V.col(k + 1).noalias() = ConjProdProd(V.col(k));
			Prod(V.col(k), V.col(k + 1).data());
			//V.col(k + 1).noalias() = Z.adjoint() * (Z * V.col(k));

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
			end = clock();
			std::cout << "time = " << JD(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
			if (error < threshold) {
				number = k;
				break;
			}
		}

		++number;
		Eigen::MatrixXcd h = H.block(0, 0, number, number);
		Eigen::VectorXcd y = h.inverse() * beta.topRows(number);
		J.noalias() += V.leftCols(number) * y;
		cout << (Z * J - b).norm() / b.norm() << endl;
		if (error < threshold) {
			break;
		}
	}
}
