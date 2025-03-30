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

	Vector3f point;
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

float RWG::TriDistence(int i, int j)
{
	if (i == j) return 0.0;
	Vector3f center1 = (points[triangles[i][0]] + points[triangles[i][1]] + points[triangles[i][2]]) / 3.0;
	Vector3f center2 = (points[triangles[j][0]] + points[triangles[j][1]] + points[triangles[j][2]]) / 3.0;
	return (center1 - center2).norm();
}

Eigen::Vector2i RWG::GetEdgeVertex(int edge_index)
{
	Vector3i& tri1 = triangles[edges[edge_index][0]], tri2 = triangles[edges[edge_index][1]];
	vector<int> count(6, 0);
	for (int j = 0; j < 3; ++j) {
		for (int k = 0; k < 3; ++k) {
			if (tri1[j] == tri2[k]) {
				count[j] = count[k + 3] = 1;
				break;
			}
		}
	}
	Vector2i edge_vertex;
	for (int j = 0; j < 3; ++j) if (count[j] == 0) edge_vertex[0] = tri1[j];
	for (int j = 0; j < 3; ++j) if (count[j + 3] == 0) edge_vertex[1] = tri2[j];
	return edge_vertex;
}

Eigen::Vector3f RWG::GetEdgeCenter(int edge_index)
{
	Vector2i edge_vertex = GetEdgeVertex(edge_index);
	Vector3i& tri1 = triangles[edges[edge_index][0]];
	for (int i = 0; i < 3; ++i) {
		if (tri1[i] == edge_vertex[0]) {
			int a = (i + 1) % 3, b = (i + 2) % 3;
			return (points[tri1[a]] + points[tri1[b]]) / 2.0;
		}
	}
}

RWG& RWG::initial(float freq)
{
	Matrix<float, 3, 3> l;
	Matrix<float, 3, 3> u;
	tri_l.reserve(triangles.size());
	tri_l_normal.reserve(triangles.size());
	tri_area.reserve(triangles.size());
	tri_normal.reserve(triangles.size());
	for (int i = 0; i < triangles.size(); ++i) {
		Vector3f& v1 = points[triangles[i][0]];
		Vector3f& v2 = points[triangles[i][1]];
		Vector3f& v3 = points[triangles[i][2]];
		l.col(0) = (v2 - v1).normalized();
		l.col(1) = (v3 - v2).normalized();
		l.col(2) = (v1 - v3).normalized();
		Vector3f n = (v2 - v1).cross((v3 - v2));
		u.col(0) = l.col(0).cross(n).normalized();
		u.col(1) = l.col(1).cross(n).normalized();
		u.col(2) = l.col(2).cross(n).normalized();
		tri_l.push_back(l);
		tri_l_normal.push_back(u);
		tri_area.push_back(n.norm() / 2.0);
		tri_normal.push_back(n.normalized());
	}

	Matrix<float, 3, 6> g;
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
		J_GL.push_back(g);
		vertex_edges.push_back(GetEdgeVertex(i));
	}
	lam = c0 / freq;
	k0 = 2.0 * pi / lam;
	gap = eps * lam;
	//cout << "size" << J_GL.size() << " " << vertex_edges.size() << endl;
	//cout << "初始化完成" << endl;
	return *this;
}

Eigen::Vector4f RWG::GetI1I2(Eigen::Vector3f r, int tri_index)
{
	Vector4f ans = Vector4f::Zero();
	for (int i = 0; i < 3; ++i) {
		Vector3f& v1 = points[triangles[tri_index][i]];
		Vector3f& v2 = points[triangles[tri_index][(i + 1) % 3]];
		float lp = (v2 - r).dot(tri_l[tri_index].col(i));
		float lm = (v1 - r).dot(tri_l[tri_index].col(i));
		float Rp = (v2 - r).norm();
		float Rm = (v1 - r).norm();
		float d = abs(tri_normal[tri_index].dot(r - v2));
		float P0 = abs((v1 - r).dot(tri_l_normal[tri_index].col(i)));
		float R0 = sqrt(d * d + P0 * P0);
		float f = log((Rp + lp) / (Rm + lm));
		float beta = atan(P0 * lp / (R0 * R0 + d * Rp)) - atan(P0 * lm / (R0 * R0 + d * Rm));
		ans.head(3) += tri_l_normal[tri_index].col(i) * (R0 * R0 * f + lp * Rp - lm * Rm);
		float P0u = (v1 - r).dot(tri_l_normal[tri_index].col(i)) / P0;
		ans[3] += P0u * (P0 * f - d * beta);
	}
	ans.head(3) *= 0.5;
	return ans;
}

std::complex<float> RWG::GetZij(int i, int j)
{
	CP C1 = Zf * k0 * 0.25f * cpdj;
	CP C2 = Zf / k0 * cpdj;
	Eigen::Matrix<float, 3, 6>& Ji = J_GL[i];
	Eigen::Matrix<float, 3, 6>& Jj = J_GL[j];
	CP Zij = CP(0.0, 0.0);
	for (int m = 0; m < 2; ++m) {
		float pmm = m ? -1.0 : 1.0;
		for (int n = 0; n < 2; ++n) {
			float pmn = n ? -1.0 : 1.0;
			float Rij = TriDistence(edges[i][m], edges[j][n]);
			float pm = m == n ? 1.0 : -1.0;
			for (int p = 0; p < GLPN; ++p) {
				//if (Rij < lam * 0.03) cout << "Zij=" << Zij << " "<<m<<" "<<n<<endl;
				if (Rij < gap) {
					//cout << i << " " << j << endl;
					Vector3f Jm = pmm * (Ji.col(m * 3 + p) - points[vertex_edges[i][m]]);
					Vector4f I1I2 = GetI1I2(Ji.col(m * 3 + p), edges[j][n]);
					Vector3cf a = C1 * Jm * pmn;
					CP b = -pm * C2 + pmn * C1 * Jm.dot(Ji.col(m * 3 + p) - points[vertex_edges[j][n]]);
					Zij += GL_points[p][0] * (I1I2.head(3).dot(a) + b * I1I2[3]) / tri_area[edges[j][n]];
					//cout << "Zij=" << Zij << " "<< I1I2.transpose()<< endl;
				}
				//cout << i << "  " << j << endl;
				for (int q = 0; q < GLPN; ++q) {
					Vector3f Jm = pmm * (Ji.col(m * 3 + p) - points[vertex_edges[i][m]]);
					Vector3f Jn = pmn * (Jj.col(n * 3 + q) - points[vertex_edges[j][n]]);
					float Rpq = (Ji.col(m * 3 + p) - Jj.col(n * 3 + q)).norm();
					float ct = cos(-Rpq * k0), st = sin(-Rpq * k0);
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