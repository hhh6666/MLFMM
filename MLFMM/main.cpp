// MLFMM.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include"MLFMM.h"
#include"base.h"

using namespace std;
using namespace Eigen;



void temp()
{
    MPIpre mpipre(1, 1);
   /* string nas_filename = "feko/ermianjiao/32lam/hh132492.nas";
    const int rwgs_num = 132492;
    const JD fines_length = 0.235 * GlobalParams.lam;*/
   /* string nas_filename = "feko/sphere/1094.nas";
    const int rwgs_num = 1094;
    const JD fines_length = 0.251 * GlobalParams.lam;*/
    /*string nas_filename = "feko/sphere/2lam/4350.nas";
    const int rwgs_num = 4350;
    const JD fines_length = 0.251 * GlobalParams.lam;*/
    /*string nas_filename = "feko/sphere/4lam/17474.nas";
    const int rwgs_num = 17474;
    const JD fines_length = 0.251 * GlobalParams.lam;*/
    /*string nas_filename = "feko/sphere/8lam/27434.nas";
    const int rwgs_num = 27434;
    const JD fines_length = 0.23 * GlobalParams.lam;*/
    string nas_filename = "feko/sphere/16lam/194452.nas";
    const int rwgs_num = 194452;
    const JD fines_length = 0.23 * GlobalParams.lam;
    RWG* old_rwg_ptr = new RWG;
    old_rwg_ptr->ReadNas(nas_filename, rwgs_num);
    OctreeRWG octree_rwg(old_rwg_ptr, fines_length, mpipre);
    octree_rwg.Fillb(90, 0);
    SpectrumPre spectrum_pre(octree_rwg.mortoncode3d.GetLevelNum(), octree_rwg.actual_L, mpipre);
    //for (int i = 0; i < 10; i++) i--;
    MatrixPre matrix_pre(octree_rwg, spectrum_pre, mpipre);
    //for (int i = 0; i < 10; i++) i--;
    MLFMM mlfmm(matrix_pre, mpipre);
    //mlfmm.ceshiP();
    mlfmm.ceshi2();
}

int main()
{
    clock_t start, end;
    start = clock();
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    temp();
    //mom_ceshi();
    //GMRES_test();
    end = clock();
    if (world_rank == 0) std::cout << world_rank << "结束" << "time = " << JD(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
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
