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
	
	/*for (int i = 0; i < triangles.size(); i++) {
		JD l1 = (points[triangles[i][1]] - points[triangles[i][0]]).norm();
		JD l2 = (points[triangles[i][2]] - points[triangles[i][1]]).norm();
		JD l3 = (points[triangles[i][0]] - points[triangles[i][2]]).norm();
		JD ca = (l1 * l1 + l2 * l2 - l3 * l3) / (2.0 * l1 * l2);
		JD cb = (l2 * l2 + l3 * l3 - l1 * l1) / (2.0 * l2 * l3);
		JD cc = (l3 * l3 + l1 * l1 - l2 * l2) / (2.0 * l3 * l1);
		JD angle_a = acos(ca) * 180.0/pi;
		JD angle_b = acos(cb) * 180.0/pi;
		JD angle_c = acos(cc) * 180.0/pi;
		if (angle_a < 10.0 || angle_b < 10.0 || angle_c < 10.0) {
			cout << angle_a << " " << angle_b << " " << angle_c << "         " << l1 << " " << l2 << " " << l3 << endl;
		}
	}*/
	//cout << "triangles.size()=" << triangles.size() << " " << triangle_num << endl;
	//cout << "points.size()=" << points.size() << " " << (triangle_num + 4) / 2 << endl;
	//cout << "edges.size()=" << edges.size() << " " << triangle_num * 1.5 << endl;
	points.shrink_to_fit();
	triangles.shrink_to_fit();
	edges.shrink_to_fit();
	//cout << "读取文件成功" << endl;
	//cout << start_point.transpose() << "       " << end_point.transpose() << endl;
	return *this;
}

JD RWG::TriDistence(int i, int j)
{
	if (i == j) return 0.0;
	VecJD3 center1 = (points[triangles[i][0]] + points[triangles[i][1]] + points[triangles[i][2]]) / JD(3.0);
	VecJD3 center2 = (points[triangles[j][0]] + points[triangles[j][1]] + points[triangles[j][2]]) / JD(3.0);
	return (center1 - center2).norm();
}


const VecJD3 RWG::GetEdgeCenter(int edge_index) const
{
	/*int tri1 = edges[edge_index][0], tri2 = edges[edge_index][1];
	VecJD3 center1 = (points[triangles[tri1][0]] + points[triangles[tri1][1]] + points[triangles[tri1][2]]) / JD(3.0);
	VecJD3 center2 = (points[triangles[tri2][0]] + points[triangles[tri2][1]] + points[triangles[tri2][2]]) / JD(3.0);
	return (center1 + center2) * JD(0.5);*/
	const Vector3i& tri1 = triangles[edges[edge_index][0]], tri2 = triangles[edges[edge_index][1]];
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
		int a = (j + 1) % 3, b = (j + 2) % 3;
		return (points[tri1[a]] + points[tri1[b]]) * JD(0.5);
	}
}

RWG& RWG::initial()
{
	Matrix<JD, 3, 3> l;
	Matrix<JD, 3, 3> u;
	//cout << 1 << endl;
	tri_l.reserve(triangles.size());
	tri_l_normal.reserve(triangles.size());
	tri_area.reserve(triangles.size());
	tri_normal.reserve(triangles.size());
	for (int i = 0; i < triangles.size(); ++i) {
		//cout << " " << i << endl;
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
		tri_area.push_back(n.norm() * JD(0.5));
		tri_normal.push_back(n.normalized());
	}
	//cout << 2 << endl;
	Eigen::Matrix<JD, 3, GLPN2> g;
	Vector2i edge_vertex;
	J_GL.reserve(edges.size());
	vertex_edges.reserve(edges.size());
	for (int i = 0; i < edges.size(); ++i) {
		for (int j = 0; j < 2; ++j) {
			Vector3i& tri = triangles[edges[i][j]];
			for (int k = 0; k < GLPN; ++k) {
				g.col(j * GLPN + k) = GL_points[k][1] * points[tri[0]] + GL_points[k][2] * points[tri[1]]
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
	//cout << 3 << endl;
	//cout << "size" << J_GL.size() << " " << vertex_edges.size() << endl;
	//cout << "初始化完成" << endl;
	return *this;
}

Eigen::Vector<JD, 4> RWG::GetI1I2(const VecJD3& r, int tri_index)
{
	Eigen::Vector<JD, 4> ans = Eigen::Vector<JD, 4>::Zero();
	//VecJD3 rr{ -0.00106007,    0.863649,    0.658491 };
	//JD alpha = 0.0;
	for (int i = 0; i < 3; ++i) {
		VecJD3& v1 = points[triangles[tri_index][i]];
		VecJD3& v2 = points[triangles[tri_index][(i + 1) % 3]];
		JD d = abs(tri_normal[tri_index].dot(r - v2));
		JD P0 = abs((v1 - r).dot(tri_l_normal[tri_index].col(i)));
		if (abs(P0) < min_eps) P0 += min_eps * 2;
		JD R02 = d * d + P0 * P0;
		JD R0 = sqrt(R02);
		
		//if (R0 < min_eps * eps2) {
		//	VecJD3 re = r + (v1 - r).dot(tri_l_normal[tri_index].col(i)) * tri_l_normal[tri_index].col(i) 
		//		//+ tri_normal[tri_index].dot(v2 - r) * tri_normal[tri_index]
		//		+ min_eps * eps2 * tri_l_normal[tri_index].col(i);
		//	//d = 0;
		//	JD P0 = abs((v1 - re).dot(tri_l_normal[tri_index].col(i)));
		//	JD R0 = sqrt(d * d + P0 * P0);
		//	JD lp = (v2 - re).dot(tri_l[tri_index].col(i));
		//	JD lm = (v1 - re).dot(tri_l[tri_index].col(i));
		//	JD Rp = (v2 - re).norm();
		//	JD Rm = (v1 - re).norm();
		//	JD f = log((Rp + lp) / (Rm + lm));
		//	JD beta = atan(P0 * lp / (R0 * R0 + d * Rp)) - atan(P0 * lm / (R0 * R0 + d * Rm));
		//	ans.head(3) += tri_l_normal[tri_index].col(i) * (R0 * R0 * f + lp * Rp - lm * Rm);
		//	JD P0u = (v1 - re).dot(tri_l_normal[tri_index].col(i)) / P0;
		//	/*if ((r - rr).norm() < 1e-4 && tri_index == 2845) {
		//		cout << f << " " << (Rm + lm) << " " << (Rp + lp) << " " << Rp<<" "<<lp << " " << min_eps * eps2 << "      " << Rm << "    " << lm << " " << R0 << " " << P0 << endl;
		//	}*/
		//	//cout << f << " " << (Rm + lm) << " " << (Rp + lp) << " " << beta << " " << min_eps * eps2 << endl;
		//	ans[3] += P0u * (P0 * f - d * beta);
		//	continue;
		//}
		
		JD lp = (v2 - r).dot(tri_l[tri_index].col(i));
		JD lm = (v1 - r).dot(tri_l[tri_index].col(i));
		JD Rp = (v2 - r).norm();
		JD Rm = (v1 - r).norm();
		/*if (lm < 0) {
			cout << Rm + lm << "  gg   " << R02 / (Rm - lm) << " " << abs(Rm + lm - R02 / (Rm - lm)) << endl;
		}*/
		/*JD uu = lp < 0 ? R02 / (Rp - lp) : Rp + lp;
		JD dd = lm < 0 ? R02 / (Rm - lm) : Rm + lm;*/
		JD temp = log((Rp + lp) / (Rm + lm));
		//JD temp = log(uu / dd);
		/*if (abs(temp - log((Rp + lp) / (Rm + lm))) > 1e-7) {
			cout <<"gg "<< abs(temp - log((Rp + lp) / (Rm + lm))) << endl;
		}*/
		JD f = isnan(temp) || isinf(temp) ? 0.0 : temp;
		//JD f = R02 < min_eps * eps2 ? 0.0 : temp;
		JD beta = atan(P0 * lp / (R02 + d * Rp)) - atan(P0 * lm / (R02 + d * Rm));
		ans.head(3) += tri_l_normal[tri_index].col(i) * (R02 * f + lp * Rp - lm * Rm);
		JD P0u = (v1 - r).dot(tri_l_normal[tri_index].col(i)) / P0;
		ans[3] += P0u * (P0 * f - d * beta);
		//alpha += P0u * (atan(lp / P0) - atan(lm / P0));
		/*if ((r-rr).norm()<1e-4&&tri_index==972) {
			cout << i << " " << beta << " " << alpha << " " << temp << " " << R0 << endl;
		}*/
	}
	ans.head(3) *= JD(0.5);
	return ans;
}

CP RWG::GetEFIEij(int i, int j)
{
	JD lam = GlobalParams.lam;
	JD k0 = GlobalParams.k0;
	JD C1 = JD(0.25);
	JD C2 = JD(1.0) / (k0 * k0);
	Eigen::Matrix<JD, 3, GLPN2>& Ji = J_GL[i];
	Eigen::Matrix<JD, 3, GLPN2>& Jj = J_GL[j];
	//JD hh = (edges_length[i] + edges_length[j]) * 0.5 / 6;
	CP Zij = CP(0.0, 0.0);
	for (int m = 0; m < 2; ++m) {
		const VecJD3& Vm = points[vertex_edges[i][m]];
		for (int n = 0; n < 2; ++n) {
			const VecJD3& Vn = points[vertex_edges[j][n]];
			JD Rij = TriDistence(edges[i][m], edges[j][n]);
			CP Zmn = CP(0.0, 0.0);
			int index = m * 2 + n;
			if (Rij < eps ) {
				CP ans1 = CP(0.0, 0.0), ans2 = CP(0.0, 0.0);
				Vector3i& trim = triangles[edges[i][m]];
				Vector3i& trin = triangles[edges[j][n]];
				for (int p = 0; p < GLS; ++p) {
					//const VecJD3& rp = Ji.col(m * GLPN + p);
					VecJD3 rp = GL_singular[p][1] * points[trim[0]] + GL_singular[p][2] * points[trim[1]] + GL_singular[p][3] * points[trim[2]];
					VecJD3 Jm = rp - Vm;
					VecJD3 d_v = tri_normal[edges[j][n]].dot(rp - Vn) * tri_normal[edges[j][n]];
					VecJD3 rho = rp - d_v;
					Vector<JD, 4> I1I2 = GetI1I2(rp, edges[j][n]);
					//CP b = -pm * C2 + pmn * C1 * Jm.dot(rp - points[vertex_edges[j][n]]);
					JD b = C1 * Jm.dot(rho - Vn) - C2;
					//CP b = C1 * (- rp.dot(d_v) - rp.dot(Vm) + d_v.dot(Vm) + rp.dot(rp) - rp.dot(Vn) + Vm.dot(Vn)) - C2;
					ans1 += GL_singular[p][0] * (C1 * I1I2.head(3).dot(Jm) + b * I1I2[3]) / tri_area[edges[j][n]];
					//ans1 += GL_points[p][0] * (C1 * I1I2.head(3).dot(Jm) / tri_area[edges[j][n]] + C1 * Jm.dot(rho - points[vertex_edges[j][n]]) * I1I2[3] / tri_area[edges[j][n]] - C2 * I1I2[3] / tri_area[edges[j][n]]) ;
					/*if (i < 5 && j < 5) {
						cout << -C2 << " " << C1 * Jm.dot(rho - points[vertex_edges[j][n]]) << "     " << I1I2.head(3).dot(a) << " " << b * I1I2[3] << endl;
					}*/
				}

				for (int q = 0; q < GLS; q++) {
					//const VecJD3& rq = Jj.col(n * GLPN + q);
					VecJD3 rq = GL_singular[q][1] * points[trin[0]] + GL_singular[q][2] * points[trin[1]] + GL_singular[q][3] * points[trin[2]];
					VecJD3 Jn = rq - Vn;
					VecJD3 d_v = tri_normal[edges[i][m]].dot(rq - Vm) * tri_normal[edges[i][m]];
					VecJD3 rho = rq - d_v;
					Vector<JD, 4> I1I2 = GetI1I2(rq, edges[i][m]);
					//CP b = -pm * C2 + pmn * C1 * Jn.dot(rq - points[vertex_edges[i][m]]);
					JD b = C1 * Jn.dot(rho - Vm) - C2;
					ans2 += GL_singular[q][0] * (C1 * I1I2.head(3).dot(Jn) + b * I1I2[3]) / tri_area[edges[i][m]];
					/*if (i == 4139 && j == 4137) {
						cout << I1I2.transpose() << " " << rq.transpose() << " " << edges[i][m] << " " << Zij << endl;
					}*/
				}
				Zmn += (ans1 + ans2) * JD(0.5);
				//Zmn += ans1;
				//if (abs(ans1-ans2)>100||isnan(abs(Zmn[index]))) {
				//	CP hhh = CP(0.0, 0.0);
				//	for (int p = 0; p < GLPN; ++p) {
				//		const VecJD3& rp = Ji.col(m * GLPN + p);

				//		VecJD3 Jm = rp - Vm;
				//		VecJD3 d_v = (tri_normal[edges[j][n]].dot(rp) - tri_normal[edges[j][n]].dot(Vn)) * tri_normal[edges[j][n]];
				//		VecJD3 rho = rp - d_v;
				//		Vector<JD, 4> I1I2 = GetI1I2(rp, edges[j][n]);
				//		//CP b = C1 * (-rp.dot(d_v) - rp.dot(Vm) + d_v.dot(Vm) + rp.dot(rp) - rp.dot(Vn) + Vm.dot(Vn)) - C2;
				//		CP b = C1 * Jm.dot(rho - Vn) - C2;
				//		hhh += GL_points[p][0] * (C1 * I1I2.head(3).dot(Jm) + b * I1I2[3]) / tri_area[edges[j][n]];
				//		//cout << rp.transpose() << "   " << edges[j][n] << endl;
				//	}
				//	cout << Zmn[index] << " " << abs(ans1-ans2) << " " << abs(hhh-ans2) <<" "<< edges[i][m] <<" "<< edges[j][n] <<endl;
				//}
				//CP ans3 = CP(0.0, 0.0);
				for (int p = 0; p < GLPN; p++) {
					VecJD3 rp = Ji.col(m * GLPN + p);
					for (int q = 0; q < GLPN; ++q) {
						VecJD3 Jm = rp - Vm;
						VecJD3 Jn = Jj.col(n * GLPN + q) - points[vertex_edges[j][n]];
						JD Rpq = (rp - Jj.col(n * GLPN + q)).norm();
						//CP G = CP(JD(-0.5) * k0 * k0 * Rpq, k0 / JD(6.0) * (Rpq * Rpq * k0 * k0 - JD(6.0)));
						CP G;
						if (Rpq < lam * 0.015) G = CP(JD(-0.5) * k0 * k0 * Rpq, k0 / JD(6.0) * (Rpq * Rpq * k0 * k0 - JD(6.0)));
						else {
							JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
							G = (CP(ct, st) - JD(1.0)) / Rpq;
						}
						Zmn += GL_points[p][0] * GL_points[q][0] * (C1 * Jm.dot(Jn) - C2) * G;
						//ans3 += GL_points[p][0] * GL_points[q][0] * (C1 * Jm.dot(Jn) - C2) * G;
					}
				}
				////Zmn[index] = CP(JD(0.0), Zmn[index].imag());
				//if (abs(ans1-ans2)>1) {
				//	cout << Zmn[index] << " "<<ans3<< endl;
				//}
			}
			else {
				for (int p = 0; p < GLPN; ++p) {
					VecJD3 rp = Ji.col(m * GLPN + p);
					for (int q = 0; q < GLPN; ++q) {
						VecJD3 Jm = rp - Vm;
						VecJD3 Jn = Jj.col(n * GLPN + q) - Vn;
						JD Rpq = (rp - Jj.col(n * GLPN + q)).norm();
						JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
						CP G = CP(ct, st) / Rpq;
						Zmn += GL_points[p][0] * GL_points[q][0] * (C1 * Jm.dot(Jn) - C2) * G;
					}

				}
				//if (i<15&&j<15) {
				//	CP ans1 = CP(0.0, 0.0), ans2 = CP(0.0, 0.0);
				//	for (int p = 0; p < GLPN; ++p) {
				//		VecJD3 rp = Ji.col(m * GLPN + p);
				//		for (int q = 0; q < GLPN; ++q) {
				//			VecJD3 Jm = (rp - points[vertex_edges[i][m]]);
				//			VecJD3 Jn = (Jj.col(n * GLPN + q) - points[vertex_edges[j][n]]);
				//			JD Rpq = (rp - Jj.col(n * GLPN + q)).norm();
				//			JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
				//			CP G = CP(ct, st) / Rpq;
				//			ans1 += GL_points[p][0] * GL_points[q][0] * (C1 * Jm.dot(Jn) - C2) * G;
				//		}
				//	}
				//	Vector3i& trim = triangles[edges[i][m]];
				//	Vector3i& trin = triangles[edges[j][n]];
				//	for (int p = 0; p < GLS; ++p) {
				//		//const VecJD3& rp = Ji.col(m * GLPN + p);
				//		VecJD3 rp = GL_singular[p][1] * points[trim[0]] + GL_singular[p][2] * points[trim[1]] + GL_singular[p][3] * points[trim[2]];
				//		VecJD3 Jm = rp - Vm;
				//		VecJD3 d_v = tri_normal[edges[j][n]].dot(rp - Vn) * tri_normal[edges[j][n]];
				//		VecJD3 r0 = rp - d_v;
				//		Vector<JD, 4> I1I2 = GetI1I2(rp, edges[j][n]);
				//		Vector<JD, 4> I1I2_num = GetI1I2_num(rp, edges[j][n]);
				//		CP b = C1 * Jm.dot(r0 - Vn) - C2;
				//		//CP b = C1 * (- rp.dot(d_v) - rp.dot(Vm) + d_v.dot(Vm) + rp.dot(rp) - rp.dot(Vn) + Vm.dot(Vn)) - C2;
				//		ans2 += GL_singular[p][0] * (C1 * Jm.dot(I1I2.head(3)) + b * I1I2[3]) / tri_area[edges[j][n]];
				//		//if(p==0)cout << I1I2.transpose() << " " << I1I2_num.transpose()* tri_area[edges[j][n]] << endl;
				//		//if (p == 0)cout << r0.cross() << endl;
				//	}
				//	for (int p = 0; p < GLPN; p++) {
				//		VecJD3 rp = Ji.col(m * GLPN + p);
				//		for (int q = 0; q < GLPN; ++q) {
				//			VecJD3 Jm = rp - points[vertex_edges[i][m]];
				//			VecJD3 Jn = Jj.col(n * GLPN + q) - points[vertex_edges[j][n]];
				//			JD Rpq = (rp - Jj.col(n * GLPN + q)).norm();
				//			JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
				//			//CP G = CP(JD(-0.5) * k0 * k0 * Rpq, k0 / JD(6.0) * (Rpq * Rpq * k0 * k0 - JD(6.0)));
				//			//CP G = CP(0.0, -k0);
				//			CP G = (CP(ct, st) - JD(1.0)) / Rpq;
				//			ans2 += GL_points[p][0] * GL_points[q][0] * (C1 * Jm.dot(Jn) - C2) * G;
				//		}
				//	}
				//	cout << "实验结果 " << Rij / lam << "    " << ans1 << " " << ans2 << "    " << abs(ans1 - ans2)/abs(ans1) << endl;
				//}
			}
			JD pm = m == n ? 1.0 : -1.0;
			Zij += pm * Zmn;
			//Zmn *= pm * edges_length[i] * edges_length[j] / (JD(4.0) * pi);
		}
	}
	//sort(Zmn.data(), Zmn.data() + 4, [](const CP& a, const CP& b) {return abs(a) < abs(b); });
	Zij *= edges_length[i] * edges_length[j] * CP(0.0, Zf * k0) / (JD(4.0) * pi);
	return Zij;
}

Eigen::Vector<JD, 4> RWG::GetI3I4(const VecJD3& r, int tri_index)
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
		if (abs(P0) < min_eps) P0 += min_eps * 2;
		JD P0u = (v1 - r).dot(tri_l_normal[tri_index].col(i)) / P0;
		//JD R0 = sqrt(d * d + P0 * P0);
		JD R02 = d * d + P0 * P0;
		JD f = log((Rp + lp) / (Rm + lm));
		if (isnan(f) || isinf(f)) {
			/*VecJD3 re = r - min_eps * 120.0 * tri_l_normal[tri_index].col(i) * P0u;
			cout << "?? " << Rm << " " << lm  << " " << (Rm + lm) << " " << f <<" "<<P0<<" "<<P0u << " "<<sqrt(R02)<<endl;
			Rp = (v2 - re).norm();
			Rm = (v1 - re).norm();
			f = log(Rp + lp) - log(Rm + lm);
			JD hh = abs((v1 - re).dot(tri_l_normal[tri_index].col(i)));
			JD hhh = (v1 - r).norm();
			cout << "??? " << Rm << " " << (Rm-hhh) << " " << (Rm + lm) << " " << f << " " << (r - re).norm() << " "<<hh<<" "<< min_eps * 100.0<<endl;*/
			f = 0;
		}
		ans.head(3) += -tri_l_normal[tri_index].col(i) * f;
		/*if (triangles[tri_index][i] == 0 && triangles[tri_index][(i + 1) % 3] == 1 || triangles[tri_index][i] == 1 && triangles[tri_index][(i + 1) % 3] == 0) {
			ans.head(3) -= -tri_l_normal[tri_index].col(i) * f;
		}*/
		
		/*if (d < min_eps) {
			JD beta = lp / (P0 * Rp) - lm / (P0 * Rm);
			ans[3] += -P0u * beta;
		}
		else {
			JD beta = atan(d * lp / (P0 * Rp)) - atan(d * lm / (P0 * Rm)) + atan(lm / P0) - atan(lp / P0);
			ans[3] += -P0u * beta / d;
		}*/
		JD beta = atan(P0 * lp / (R02 + d * Rp)) - atan(P0 * lm / (R02 + d * Rm));
		
		ans[3] += P0u * beta;
	}
	return ans;
}

CP RWG::GetMFIEij(int i, int j)
{
	JD lam = GlobalParams.lam;
	JD k0 = GlobalParams.k0;
	Eigen::Matrix<JD, 3, GLPN2>& Ji = J_GL[i];
	Eigen::Matrix<JD, 3, GLPN2>& Jj = J_GL[j];
	CP Zij1 = CP(0.0, 0.0), Zij2 = CP(0.0, 0.0);
	
	for (int m = 0; m < 2; ++m) {
		int trim_index = edges[i][m];
		for (int n = 0; n < 2; ++n) {
			int trin_index = edges[j][n];
			JD pm = m == n ? 1.0 : -1.0;
			if (trim_index == trin_index) {
				CP temp = CP(0.0, 0.0);
				for (int p = 0; p < GLPN; ++p) {
					VecJD3 Jm = (Ji.col(m * GLPN + p) - points[vertex_edges[i][m]]);
					VecJD3 Jn = (Ji.col(m * GLPN + p) - points[vertex_edges[j][n]]);
					temp += GL_points[p][0] * (Jm.dot(Jn));
				}
				Zij2 += pm * temp / tri_area[trim_index];
				continue;
			}
			
			JD Rij = TriDistence(trim_index, trin_index);
			JD ans1 = 0, ans2 = 0;
			if (Rij < lam * 0.4) {
				//if (i == j)cout << "here " << i << " "<<m<<" "<<n<<" "<< Rij<<" "<< lam * 0.02<<endl;
				Vector3i& trim = triangles[trim_index];
				Vector3i& trin = triangles[trin_index];
				for (int p = 0; p < GLS; ++p) {
					VecJD3 rp = GL_singular[p][1] * points[trim[0]] + GL_singular[p][2] * points[trim[1]] + GL_singular[p][3] * points[trim[2]];
					VecJD3 Jm = (rp - points[vertex_edges[i][m]]);
					JD d = tri_normal[trin_index].dot(rp - points[vertex_edges[j][n]]);
					JD pmd = abs(d) < min_eps ? JD(0.0) : d / abs(d);
					VecJD3 Rn = rp - points[vertex_edges[j][n]];
					VecJD3 temp1 = VecJD3::Zero(), temp2 = VecJD3::Zero();
					Vector<JD, 4> I1I2 = GetI1I2(rp, trin_index);
					Vector<JD, 4> I3I4 = GetI3I4(rp, trin_index);
					/*Vector<JD, 4> I3I4_num = GetI3I4_num(rp, trin_index);
					Vector<JD, 4> I1I2_num = GetI1I2_num(rp, trin_index);*/
					temp1 = (I3I4.head(3) - tri_normal[trin_index] * pmd * I3I4[3]) / tri_area[trin_index];
					temp2 = k0 * k0 * JD(0.5) * (I1I2.head(3) - tri_normal[trin_index] * d * I1I2[3]) / tri_area[trin_index];
					ans1 += GL_singular[p][0] * Jm.dot(tri_normal[trim_index].cross(Rn.cross(temp2 + temp1)));
					//cout <<p<<" " << "       " << Zij1 << " " << temp1.transpose() << " " << I3I4.transpose() << " "<<pmd<<endl;
					/*if (i == 129227 && j == 129231) {
						cout << p << " " << "       " << Zij1 << " " << I1I2.transpose() << " " << I3I4.transpose() << endl;
					}*/
				}

				//for (int q = 0; q < GLS; ++q) {
				//	VecJD3 rq = GL_singular[q][1] * points[trin[0]] + GL_singular[q][2] * points[trin[1]] + GL_singular[q][3] * points[trin[2]];
				//	VecJD3 Jn = (rq - points[vertex_edges[j][n]]);
				//	JD d = tri_normal[trim_index].dot(rq - points[vertex_edges[i][m]]);
				//	JD pmd = abs(d) < min_eps ? JD(0.0) : d / abs(d);
				//	VecJD3 Rm = rq - points[vertex_edges[i][m]];
				//	VecJD3 temp1 = VecJD3::Zero(), temp2 = VecJD3::Zero();
				//	Vector<JD, 4> I1I2 = GetI1I2(rq, trim_index);
				//	Vector<JD, 4> I3I4 = GetI3I4(rq, trim_index);
				//	/*Vector<JD, 4> I3I4_num = GetI3I4_num(rp, trin_index);
				//	Vector<JD, 4> I1I2_num = GetI1I2_num(rp, trin_index);*/
				//	temp1 = (I3I4.head(3) - tri_normal[trim_index] * pmd * I3I4[3]) / tri_area[trim_index];
				//	temp2 = k0 * k0 * JD(0.5) * (I1I2.head(3) - tri_normal[trim_index] * d * I1I2[3]) / tri_area[trim_index];
				//	ans2 += GL_singular[q][0] * Jn.dot(tri_normal[trin_index].cross(Rm.cross(temp2 + temp1)));
				//	//cout <<p<<" " << "       " << Zij1 << " " << temp1.transpose() << " " << I3I4.transpose() << " "<<pmd<<endl;
				//}
				//Zij1 += pm * (ans1 + ans2) * 0.5;
				Zij1 += pm * ans1;
				for (int p = 0; p < GLPN; ++p)
				{
					VecJD3 rp = Ji.col(m * GLPN + p);
					VecJD3 Jm = rp - points[vertex_edges[i][m]];
					VecJD3 Rn = rp - points[vertex_edges[j][n]];
					VecCP3 temp = VecCP3::Zero();
					for (int q = 0; q < GLPN; ++q) {
						//VecJD3 Jn = pmn * (Jj.col(n * 3 + q) - points[vertex_edges[j][n]]);
						VecJD3 Jn = Jj.col(n * GLPN + q) - rp;
						JD Rpq = (rp - Jj.col(n * GLPN + q)).norm();
						CP G;
						if (Rpq > lam * 0.05) {
							JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
							G = ((JD(1.0) + cpdj * k0 * Rpq) * CP(ct, st) - JD(1.0)) / (Rpq * Rpq * Rpq) - k0 * k0 * JD(0.5) / Rpq;
						}
						else G = -k0 * k0 * k0 * CP(k0 * Rpq * JD(0.125) * (JD(1.0) - k0 * k0 * Rpq * Rpq / JD(18.0)), (JD(1.0) - k0 * k0 * Rpq * Rpq * JD(0.1)) / JD(3.0));
						
						temp += GL_points[q][0] * G * Jn;
					}
					Zij1 += pm * GL_points[p][0] * Jm.dot(tri_normal[trim_index].cross(Rn.cross(temp)));
				}
			}
			else {
				for (int p = 0; p < GLPN; ++p) {
					VecJD3 Jm = (Ji.col(m * GLPN + p) - points[vertex_edges[i][m]]);
					for (int q = 0; q < GLPN; ++q) {
						VecJD3 Jn = (Jj.col(n * GLPN + q) - points[vertex_edges[j][n]]);
						JD Rpq = (Ji.col(m * GLPN + p) - Jj.col(n * GLPN + q)).norm();
						JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
						CP G = (JD(1.0) + cpdj * k0 * Rpq) * CP(ct, st) / (Rpq * Rpq * Rpq);
						VecJD3 temp = tri_normal[trim_index].cross((Ji.col(m * GLPN + p) - Jj.col(n * GLPN + q)).cross(Jn));
						Zij1 += pm * GL_points[p][0] * GL_points[q][0] * (Jm.dot(temp)) * G;
					}
				}
				/*if (Rij>lam*0.15) {
					cout << "开始实验 " << endl;
					CP ans1 = CP(0, 0);
					for (int p = 0; p < GLPN; ++p) {
						VecJD3 Jm = (Ji.col(m * GLPN + p) - points[vertex_edges[i][m]]);
						for (int q = 0; q < GLPN; ++q) {
							VecJD3 Jn = (Jj.col(n * GLPN + q) - points[vertex_edges[j][n]]);
							JD Rpq = (Ji.col(m * GLPN + p) - Jj.col(n * GLPN + q)).norm();
							JD ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
							CP G = (JD(1.0) + cpdj * k0 * Rpq) * CP(ct, st) / (Rpq * Rpq * Rpq);
							VecJD3 temp = tri_normal[trim_index].cross((Ji.col(m * GLPN + p) - Jj.col(n * GLPN + q)).cross(Jn));
							ans1 += GL_points[p][0] * GL_points[q][0] * (Jm.dot(temp)) * G;
						}
					}
					CP ans2 = CP(0, 0);
					Vector3i& trim = triangles[trim_index];
					Vector3i& trin = triangles[trin_index];
					for (int p = 0; p < GLS; ++p) {
						VecJD3 rp = GL_singular[p][1] * points[trim[0]] + GL_singular[p][2] * points[trim[1]] + GL_singular[p][3] * points[trim[2]];
						VecJD3 Jm = (rp - points[vertex_edges[i][m]]);
						JD d = tri_normal[trin_index].dot(rp - points[vertex_edges[j][n]]);
						VecJD3 Rn = rp - points[vertex_edges[j][n]];
						VecJD3 temp1 = VecJD3::Zero(), temp2 = VecJD3::Zero();
						Vector<JD, 4> I1I2 = GetI1I2(rp, trin_index);
						Vector<JD, 4> I3I4 = GetI3I4(rp, trin_index);
						Vector<JD, 4> I3I4_num = GetI3I4_num(rp, trin_index);
						Vector<JD, 4> I1I2_num = GetI1I2_num(rp, trin_index);
						temp1 = (I3I4.head(3) - tri_normal[trin_index] * d / abs(d) * I3I4[3]) / tri_area[edges[j][n]];
						temp2 = k0 * k0 * JD(0.5) * (I1I2.head(3) - tri_normal[trin_index] * d * I1I2[3]) / tri_area[edges[j][n]];
						ans2 += GL_singular[p][0] * Jm.dot(tri_normal[trim_index].cross(Rn.cross(temp2 + temp1)));
						cout << I3I4_num.transpose() * tri_area[edges[j][n]] << "             " << I3I4.transpose() << "    "<<d/lam<<endl;
						cout << I1I2_num.transpose() * tri_area[edges[j][n]] << "             " << I1I2.transpose() << "    " << d / lam << endl;
					}
					cout << "实验结果 " << Rij / lam << "    " << ans1 << " " << ans2 << "    " << abs(ans1 - ans2) <<endl;
				}*/
			}
		}
	}
	Zij1 *= edges_length[i] * edges_length[j] / (JD(16.0) * pi);
	Zij2 *= edges_length[i] * edges_length[j] / JD(8.0);
	return Zij1 + Zij2;
}

CP RWG::GetZij(int i, int j)
{
	CP temp;
	if (abs(alpha - 0.0) < min_eps)temp = Zf * GetMFIEij(i, j);
	else if (abs(alpha - 1.0) < min_eps)temp = GetEFIEij(i, j);
	else temp = alpha * GetEFIEij(i, j) + Zf * JD(1.0 - alpha) * GetMFIEij(i, j);
	return temp;
}

CP RWG::Getbi(int i)
{
	CP temp;
	if (abs(alpha - 0.0) < min_eps)temp = GetHbi(i);
	else if (abs(alpha - 1.0) < min_eps)temp = GetEbi(i);
	else temp = alpha * GetEbi(i) + JD(1.0 - alpha) * GetHbi(i);
	return temp;
}

void MOM::Fillb()
{
	b.setZero(rwg.edges.size());
	for (int i = 0; i < rwg.edges.size(); i++) {
		b[i] = rwg.Getbi(i);
	}
}


CP RWG::GetEbi(int i)
{
	CP bi(0, 0);
	Eigen::Matrix<JD, 3, GLPN2>& Ji = J_GL[i];
	for (int m = 0; m < 2; ++m) {
		JD pmm = m ? -1.0 : 1.0;
		for (int p = 0; p < GLPN; ++p) {
			VecJD3 Jm = pmm * (Ji.col(m * GLPN + p) - points[vertex_edges[i][m]]);
			JD rpr = -GlobalParams.k0 * Ji.col(m * GLPN + p).dot(k_unit);
			CP ejkr = CP(cos(rpr), sin(rpr));
			bi += GL_points[p][0] * Jm.dot(Ep) * ejkr;
		}
	}
	return bi * edges_length[i] * JD(0.5);
}
CP RWG::GetHbi(int i)
{
	CP bi(0, 0);
	Eigen::Matrix<JD, 3, GLPN2>& Ji = J_GL[i];
	for (int m = 0; m < 2; ++m) {
		int trim = edges[i][m];
		JD pmm = m ? -1.0 : 1.0;
		for (int p = 0; p < GLPN; ++p) {
			VecJD3 Jm = pmm * (Ji.col(m * GLPN + p) - points[vertex_edges[i][m]]);
			JD rpr = -GlobalParams.k0 * Ji.col(m * GLPN + p).dot(k_unit);
			CP ejkr = CP(cos(rpr), sin(rpr));
			bi += GL_points[p][0] * Jm.dot(tri_normal[trim].cross(Hp)) * ejkr;
		}
	}
	return bi * edges_length[i] * JD(0.5);
}

Eigen::Vector<CP, 2> MOM::FarField(JD theta, JD phi)
{
	JD ct = cos(theta * rad), st = sin(theta * rad), cp = cos(phi * rad), sp = sin(phi * rad);
	VecJD3 r = VecJD3(st * cp, st * sp, ct);
	VecCP3 e = VecCP3::Zero();
	Eigen::Matrix<JD, 2, 3> T;
	T << ct * cp, ct* sp, -st,
		-sp, cp, 0;
	for (int i = 0; i < rwg.edges.size(); i++) {
		Eigen::Matrix<JD, 3, GLPN2>& Ji = rwg.J_GL[i];
		for (int m = 0; m < 2; ++m) {
			JD pmm = m ? -1 : 1;
			for (int p = 0; p < GLPN; ++p) {
				VecJD3 Jm = pmm * (Ji.col(m * GLPN + p) - rwg.points[rwg.vertex_edges[i][m]]);
				JD inrpr = GlobalParams.k0 * Ji.col(m * GLPN + p).dot(r);
				JD ct = cos(inrpr), st = sin(inrpr);
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
		for (int j = 0; j < size; j++) {
			Z(j, k) = rwg.GetZij(j, k);
			/*if (j<=200&&k==0) {
				cout << j << " " << k << " "<< Z(j, k)<<endl;
			}*/
		}
	}
	/*for (int k = 0; k < size; k++) {
		for (int j = k; j < size; j++) {
			Z(k, j) = Z(j, k);
		}
	}*/

	//Z_index.resize(N+1);
	//lu_list.resize(N);
	//for (int i = 0; i < N; ++i) {
	//	Z_index[i] = i * size / N;
	//	cout << Z_index[i] << endl;
	//}
	//Z_index[N] = size;
	//for (int i = 0; i < N; ++i) {
	//	int block_size = Z_index[i + 1] - Z_index[i];
	//	Eigen::PartialPivLU<MatCP> Zinv(Z.block(Z_index[i], Z_index[i], block_size, block_size));
	//	lu_list[i] = Zinv;
	//	//if(i==0)cout << Zinv.solve(Z.block(Z_index[i], Z_index[i], block_size, block_size)).block(0,0,5,5) << endl;
	//}
}

void MOM::SelfProd(VecCP& b, CP* b_str)
{
	Map<VecCP> b_map(b_str, b.size());
	b_map = b;
	/*for (int i = 0; i < Z_index.size() - 1; i++) {
		int block_size = Z_index[i + 1] - Z_index[i];
		b_map.segment(Z_index[i], block_size) = lu_list[i].solve(b.segment(Z_index[i], block_size));
	}*/
}

void MOM::Prod(const VecCP& x, CP* b_str)
{
	Map<VecCP> b_map(b_str, x.size());
	b_temp.setZero();
	//for (int i = 0; i < Z_index.size() - 1; i++) {
	//	int block_size_i = Z_index[i + 1] - Z_index[i];
	//	//b_temp.segment(Z_index[i], block_size_i).noalias() += Z.block(Z_index[i], Z_index[i], block_size_i, block_size_i) * x.segment(Z_index[i], block_size_i);
	//	
	//	for (int j = 0; j < Z_index.size() - 1; ++j) {
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
	b_temp.resize(b.size());
	VecCP Zb(b.size());
	//Zb = Z.adjoint() * b;
	SelfProd(b, Zb.data());
	cout << "Zb done" << " " << Zb.squaredNorm() << endl;

	//Eigen::MatCP ZZ = Z.adjoint() * Z;
	clock_t start, end;
	start = clock();
	J.setZero();
	JD Zb_norm = Zb.norm();
	size_t restart_num = 1;
	size_t iterations_max = 1000;
	size_t lieshu = J.rows();
	JD threshold = 1e-3;
	VecCP Hk = VecCP::Zero(iterations_max);
	for (int j = 0; j < restart_num; ++j) {
		VecCP r = Zb;
		//if (j > 0) r -= ConjProdProd(J);
		VecCP c = VecCP::Zero(iterations_max);
		VecCP s = VecCP::Zero(iterations_max);
		VecCP e1 = VecCP::Zero(iterations_max);
		VecCP beta = VecCP::Zero(iterations_max);
		MatCP H = MatCP::Zero(iterations_max, iterations_max);
		MatCP V = MatCP::Zero(lieshu, iterations_max);
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
			JD tao = sqrt(norm(H(k, k)) + norm(H(k + 1, k)));

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
		MatCP h = H.block(0, 0, number, number);
		VecCP y = h.inverse() * beta.topRows(number);
		J.noalias() += V.leftCols(number) * y;
		cout << (Z * J - b).norm() / b.norm() << endl;
		if (error < threshold) {
			break;
		}
	}
}
