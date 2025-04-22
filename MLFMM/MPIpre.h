#ifndef MPIPRE_H
#define MPIPRE_H

#include <mpi.h>
#include <iostream>
#include <vector>

struct CommObj
{
	int rank_exchange = MPI_PROC_NULL;
	int rank_up = MPI_PROC_NULL, rank_down = MPI_PROC_NULL;
};

class MPIpre
{
	int rank = 0;
	int size = 1;
	int BP_end = 1;
	int HSP_end = BP_end;
	int BP_size = 1;
	std::vector<int> process_num;//当前层进程组数量
	std::vector<int> process_index;//本地进程在每一层进程组的位置
	std::vector<std::vector<int>> transfers_index; //每一个HSP层和本地进程存相同平面波进程的索引
	std::vector<CommObj> HSP_com;
	std::vector<int> BP_com;
public:
	MPIpre() : MPIpre(0, 1) {}
	MPIpre(int BP_end, int size_BP) : BP_end(BP_end), BP_size(size_BP) {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		int temp_size = size;
		while (temp_size / (1 << (HSP_end - BP_end)) > BP_size) ++HSP_end;
		this->GetProcessOrder();
	}
	const int Get_BP_com(int index) const {
		return BP_com[index];
	}
	bool is_PWP(int level_index) const {
		return level_index >= HSP_end;
	}
	const int Get_BP_end() const {
		return BP_end;
	}
	int Get_HSP_end() { return HSP_end; }
	bool is_BP(int level_index)const {
		return level_index <= BP_end;
	}
	bool is_HSP(int level_index)const {
		return level_index < HSP_end && level_index > BP_end;
	}
	void GetProcessOrder();
	//当前进程在每一层进程组的索引
	int GetProcessIndex(int level_index) const {
		if (level_index < BP_end) {
			return 0;
		}
		return level_index < HSP_end ? process_index[HSP_end - level_index] : rank; 
	}
	int GetProcessNum(int level_index) const { //每一层进程组的进程数
		if (level_index < BP_end) {
			return 1;
		}
		return level_index < HSP_end ? process_num[HSP_end - level_index] : size; 
	}
	int GetRankUp(int level_index) const {
		if (level_index < BP_end || (level_index >= HSP_end && rank == 0)) return MPI_PROC_NULL;
		else return level_index < HSP_end ? HSP_com[HSP_end - level_index].rank_up : rank - 1;
	}
	int GetRankDown(int level_index) const {
		if(level_index < BP_end || (level_index >= HSP_end && rank == size - 1)) return MPI_PROC_NULL;
		else return level_index < HSP_end ? HSP_com[HSP_end - level_index].rank_down : rank + 1;
	}
	int GetRankExchange(int level_index) const {return HSP_com[HSP_end - level_index].rank_exchange;}
	std::vector<int>& GetTransferProcess(int level_index) { 
		if (level_index >= HSP_end) return transfers_index[0];
		return level_index <= BP_end ? transfers_index[transfers_index.size() - 1] : transfers_index[HSP_end - level_index];
	}
	int GetSize()const{return size;}
	int GetRank()const{return rank;}
};

#endif
