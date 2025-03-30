#include "MPIpre.h"
#include <algorithm>
using namespace std;

void MPIpre::GetProcessOrder()
{
	int cluster_size = size;
	std::vector<std::vector<int> > process;
	//cout << "rank=" << rank << endl;
	int level_num = 0;
	while (cluster_size >= BP_size)
	{
		std::vector<int> process_now(size);
		std::vector<int> plane_wave_now(size / cluster_size);
		CommObj comm_obj;
		if (cluster_size == size) {
			for (int i = 0; i < size; i++) {
				process_now[i] = i;
			}
			process.push_back(process_now);
			process_index.push_back(rank);
			process_num.push_back(cluster_size);
			if (rank < cluster_size - 1) {
				comm_obj.rank_down = process[level_num][rank + 1];
			}
			if (rank > 0) {
				comm_obj.rank_up = process[level_num][rank - 1];
			}
			HSP_com.push_back(comm_obj);

			plane_wave_now[0] = rank;
			transfers_index.push_back(plane_wave_now);
		}
		else {
			int rank_index = 0;//�����ڵ�ǰ�ܽ����������
			for (int i = 0; i < size; i++) {
				//cout << i << endl;
				int a = i / cluster_size;//�ڼ���������
				int b = i % cluster_size;//��������ڼ�������
				int c = a % 2 == 0 ? b * 2 : b * 2 + 1; //�Ӹ��׽�����ڼ�����������
				int index = (a / 2) * (cluster_size * 2) + c;//��ǰ�����ڸ������������
				process_now[i] = process[level_num - 1][index];
				int d = a % 2 == 0 ? b * 2 + 1 : b * 2;
				int exchange_index = (a / 2) * (cluster_size * 2) + d;//���������������
				//cout << i << " " << process_now[i] << " " << rank << " " << a << " " << b << " " << c << " " << d << " " << exchange_index << endl;
				//cout<< process[level_num - 1].size() << endl;
				if (process_now[i] == rank) {
					comm_obj.rank_exchange = process[level_num - 1][exchange_index];
					process_num.push_back(cluster_size);
					process_index.push_back(b);
					rank_index = i;
				}
			}
			process.push_back(process_now);
			int b = rank_index % cluster_size;//�����ڵ�ǰ������ڼ������̣����õ�������BP��
			for (int i = 0; i < size / cluster_size; i++) {
				plane_wave_now[i] = b + i * cluster_size;
			}
			sort(plane_wave_now.begin(), plane_wave_now.end());
			transfers_index.push_back(plane_wave_now);

			if (BP_size > 1 && cluster_size == BP_size) {
				BP_com.reserve(cluster_size);
				for (int i = 0; i < cluster_size; i++) {
					BP_com.push_back(process_now[i + (rank_index / cluster_size) * cluster_size]);
				}
			}
			if (b < cluster_size - 1) {
				comm_obj.rank_down = process[level_num][rank_index + 1];
			}
			if (b > 0) {
				comm_obj.rank_up = process[level_num][rank_index - 1];
			}
			HSP_com.push_back(comm_obj);
		}
		
		++level_num;
		cluster_size /= 2;
	}
	/*if (rank == 2) {
		cout << "���̷���" << process.size() << endl;
		for (auto& i : process) {
			for (auto& j : i) {
				cout << j << " ";
			}
			cout << endl;
		}
		cout << "�ߴ�" << process_num.size() << endl;
		for (auto& i : process_num) {
			cout << i << " ";
		}
		cout << endl;
		cout << "����" << process_index.size() << endl;
		for (auto& i : process_index) {
			cout << i << " ";
		}
		cout << endl;
		for (auto& i : HSP_com) {
			cout << i.rank_exchange << " " << i.rank_up << " " << i.rank_down << endl;
		}
	}*/
	
}

