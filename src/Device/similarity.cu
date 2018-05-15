/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

Appagato is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include "Device/approx_subgraph_iso.cuh"
#include "XLib.hpp"
#include "Kernels/BFSSimilarityQueryKernel.cu"
#include "Kernels/BFSSimilarityTargetKernel.cu"

namespace appagato {
namespace device {

void approx_subgraph_iso::similarity(int* devSimilarity_matrix) {
	dim3 dimBlock(16, 8);
	dim3 dimGridA( Div(MAX_LABELS * MAX_LABELS, dimBlock.x), Div(Query.V, dimBlock.y) );
	dim3 dimGridB( Div(Target.V, dimBlock.x), Div(Target.V, dimBlock.y) );
	cuda_util::fill <<< dimGridA, dimBlock >>>
            (devMarked_matrix_query, Query.V, MAX_LABELS * MAX_LABELS, false, pitchQueryBFS);
	cuda_util::fill <<< dimGridB, dimBlock >>>
            (devMarked_matrix_target, Target.V, Target.V, (mark_t) 0, pitchMarkedBFS / sizeof(mark_t));

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("BFS Similarity Kernel Init"));
    __ENABLE(PRINT_INFO, __PRINT("-> BFS Similarity Kernel Query\t\tMax Edges: "
                                    << REMAIN_SMEM_ITEMS_QUERY << std::endl));

	bfsSimilarityQueryKernel<<< Query.V, BLOCKDIM_SIM_QUERY >>>
				(devNodeQuery, devEdgeQuery, devLabelQuery, devMarked_matrix_query, pitchQueryBFS, Query.V);

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("BFS Similarity Query Kernel"));
	__ENABLE(CHECK_RESULT, check_query_similarity(devMarked_matrix_query););
    __ENABLE(PRINT_INFO, __PRINT("-> BFS Similarity Kernel Target\t\tMax Nodes: "
                                    << MAX_TARGET_FRONTIER_SIZE));
    if (!LOW_MEMORY)
        __ENABLE(PRINT_INFO, __PRINT("\t\tMax Edges: " << MAX_TARGET_FRONTIER2_SIZE));
    __ENABLE(PRINT_INFO, __PRINT(""));

	bfsSimilarityTargetKernel	<<< Target.V, BLOCKDIM_SIM_TARGET, SMem_Per_Block<char, BLOCKDIM_SIM_TARGET>::value >>>
					(devNodeTarget, devEdgeTarget, devLabelTarget, devDegreeTarget,
					devMarked_matrix_target, devMarked_matrix_query, devSimilarity_matrix,
					pitchMarkedBFS / sizeof(mark_t), pitchQueryBFS / sizeof(bool), pitchBFSSIM / sizeof(int), Query.V);

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("BFS Similarity Target Kernel"));
    __ENABLE(CHECK_RESULT, check_target_similarity(devMarked_matrix_query, devSimilarity_matrix););
	cudaFree(devMarked_matrix_target);
	cudaFree(devMarked_matrix_query);
}


void approx_subgraph_iso::check_query_similarity(const bool* devMarked_matrix_query) {
    bool* Marked_matrix_query_HOST = new bool[MAX_LABELS * MAX_LABELS * Query.V]();

	q_node_t* Distance = new q_node_t[Query.V];
	std::fill(Distance, Distance + Query.V, INF_QNODE);

    fast_queue::Queue<q_node_t> Queue(Query.V);
    fast_queue::Queue<q_node_t> QueueTMP(Query.V);
    for (int source = 0; source < Query.V; ++source)
        BFS_Similarity<false>(Query, Marked_matrix_query_HOST, (q_node_t) source, Distance, Queue, QueueTMP);

    //--------------------------------------------------------------------------

    bool* Marked_matrix_query_DEVICE = new bool[MAX_LABELS * MAX_LABELS * Query.V];
	cudaMemcpy2D(Marked_matrix_query_DEVICE, MAX_LABELS * MAX_LABELS * sizeof (bool),
                devMarked_matrix_query, pitchQueryBFS, MAX_LABELS * MAX_LABELS * sizeof (bool), Query.V,
                cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------

    /*for (int k = 0; k < Query.V; k++) {
        for (int i = 0; i < MAX_LABELS; i++) {
            for (int j = 0; j < MAX_LABELS; j++) {
                if (Marked_matrix_query_DEVICE[MAX_LABELS * MAX_LABELS * k + i * MAX_LABELS + j] !=
                    Marked_matrix_query_HOST[MAX_LABELS * MAX_LABELS * k + i * MAX_LABELS + j]) {
                        std::cout << "error " << i << " " << j << "   " << k << "   -- "
                                    << Marked_matrix_query_HOST[MAX_LABELS * MAX_LABELS * k + i * MAX_LABELS + j] << " "
                                    << Marked_matrix_query_DEVICE[MAX_LABELS * MAX_LABELS * k + i * MAX_LABELS + j] << std::endl;
                        std::exit(1);
                }
            }
        }
    }*/

	if( !std::equal(Marked_matrix_query_HOST, Marked_matrix_query_HOST + MAX_LABELS * MAX_LABELS * Query.V, Marked_matrix_query_DEVICE) )
		__ERROR("Wrong Query BFS Cost");

	std::cout << "\t<> Query BFS Similarity CORRECT" << std::endl << std::endl;
    delete Marked_matrix_query_HOST;
	delete Marked_matrix_query_DEVICE;
    delete Distance;
}



void approx_subgraph_iso::check_target_similarity(const bool* devMarked_matrix_query, const int* devSimilarity_matrix) {
    bool* Marked_matrix_query_DEVICE = new bool[MAX_LABELS * MAX_LABELS * Query.V];
	cudaMemcpy2D(Marked_matrix_query_DEVICE, MAX_LABELS * MAX_LABELS * sizeof (bool),
                devMarked_matrix_query, pitchQueryBFS, MAX_LABELS * MAX_LABELS * sizeof (bool), Query.V,
                cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------

    int* Similarity_matrix_HOST = new int[Query.V * Target.V]();

    q_node_t* Distance = new q_node_t[Target.V];
	std::fill(Distance, Distance + Target.V, INF_QNODE);

    fast_queue::Queue<t_node_t> Queue(Target.V);
    fast_queue::Queue<t_node_t> QueueTMP(Target.V);
    for (int source = 0; source < Target.V; ++source)
        BFS_Similarity<true>(Target, Marked_matrix_query_DEVICE, (t_node_t) source,
                            Distance, Queue, QueueTMP, Similarity_matrix_HOST);

    //--------------------------------------------------------------------------

    int* Similarity_matrix_DEVICE = new int[Query.V * Target.V];
    cudaMemcpy2D(Similarity_matrix_DEVICE, Target.V * sizeof (int),
                devSimilarity_matrix, pitchBFSSIM, Target.V * sizeof (int), Query.V,
                cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------

    if( !std::equal(Similarity_matrix_HOST, Similarity_matrix_HOST + Query.V * Target.V, Similarity_matrix_DEVICE) )
        __ERROR("Wrong Target BFS Cost");

    std::cout << "\t<> Target BFS Similarity CORRECT" << std::endl << std::endl;
    delete Distance;
    delete Similarity_matrix_DEVICE;
    delete Marked_matrix_query_DEVICE;
}

} //@device
} //@appagato
