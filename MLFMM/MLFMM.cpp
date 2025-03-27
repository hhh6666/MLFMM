// MLFMM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include"Octree.h"
#include"base.h"
#include <iostream>

using namespace std;
using namespace Eigen;

struct Parameters
{
    float freq;
    float lam;
    float waveNumber;
    Parameters(float freq) :freq(freq), lam(c0 / freq), waveNumber(2 * pi / lam) {}
};

void temp()
{
    MPIpre mpipre(2, 2);
    Parameters params(1.5e9);
    string nas_filename = "feko/ermianjiao/32lam/hh132492.nas";
    string efe_filename = "feko/ermianjiao/32lam/er_NearField1.efe";
    int sample_num = 16 * 159;//0.2501
    RWG* old_rwg_ptr = new RWG;
    old_rwg_ptr->ReadNas(nas_filename, 132492);
    OctreeRWG octree_rwg(old_rwg_ptr, 0.47 * params.lam, mpipre);
}

int main()
{
    clock_t start, end;
    start = clock();
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    temp();
    end = clock();
    if (world_rank == 0) std::cout << world_rank << "结束" << "time = " << float(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    MPI_Finalize();
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
