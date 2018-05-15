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
#pragma once
#include "XLib.hpp"
#include "../../../config.h"
#include "types.hpp"
using namespace primitives;

namespace appagato {
namespace device {

const int TEMP_NODE_SIM_TARGET_POS = sizeof(int);   //temp_position = total_position + 1
const int FRONTIER_SIM_TARGET_POS = TEMP_NODE_SIM_TARGET_POS +                       // total size
                                    (BLOCKDIM_SIM_TARGET / WARP_SIZE) * sizeof(int); // temp size

const int MAX_TARGET_FRONTIER_SIZE = (SMem_Per_Block<char, BLOCKDIM_SIM_TARGET>::value - FRONTIER_SIM_TARGET_POS) / sizeof(t_node_t);
//NO LOW MEMORY
const int MAX_TARGET_FRONTIER2_SIZE = (SMem_Per_Block<char, BLOCKDIM_SIM_TARGET>::value - FRONTIER_SIM_TARGET_POS) / sizeof(t_node_t2);

extern __shared__ char SMem[];
//------------------------------------------------------------------------------

template <int VW_SIZE>
__device__ __forceinline__ void NodeFrontierExpansion (	const t_edge_t* __restrict__ devNode,
													const t_node_t* __restrict__ devEdge,
													mark_t* __restrict__         devMarked,
													int&                         FrontierSizeNode);

template <int VW_SIZE>
__device__ __forceinline__ void NodeFrontier_to_Score (	const t_edge_t* __restrict__ devNode,
											const t_node_t* __restrict__ devEdge,
											const int                    FrontierSizeNode,
											const label_t* __restrict__  devLabel,
											const t_node_t* __restrict__ devDegree,
											const bool* __restrict__     devQueryBFS,
											int* __restrict__            devBFSSIM,
											const int                    pitchQueryBFS,
											const int                    pitchBFSSIM,
											const int                    query_nodes);

//------------------------------------------------------------------------------

__global__ void bfsSimilarityTargetKernel (	const t_edge_t* __restrict__ devNode,
											const t_node_t* __restrict__ devEdge,
											const label_t* __restrict__  devLabel,
											const t_node_t* __restrict__ devDegree,
											mark_t* __restrict__         devMarked,
											const bool* __restrict__     devQueryBFS,
											int* __restrict__            devBFSSIM,
											const int                    pitchMarked,
											const int                    pitchQueryBFS,
											const int                    pitchBFSSIM,
											const int                    query_nodes) {

    t_node_t* FrontierNode = (t_node_t*) (SMem + FRONTIER_SIM_TARGET_POS);
	if (threadIdx.x == 0)
		FrontierNode[0] = blockIdx.x;             // set SOURCE

	devMarked += blockIdx.x * pitchMarked;
	if (threadIdx.x == 0)
		devMarked[blockIdx.x] = true;            // reset source
	__syncthreads();

	int FrontierSizeNode = 1;
	for (int level = 1; FrontierSizeNode > 0 && level < BFS_SIMILARITY_DEEP; level++) {

		const int logSize = logValue<BLOCKDIM_SIM_TARGET, MIN_VW_SIM_TARGET, MAX_VW_SIM_TARGET>(FrontierSizeNode);
        #define fun(a)	NodeFrontierExpansion <(a)> (devNode, devEdge, devMarked, FrontierSizeNode);

		def_SWITCH(logSize);
        #undef fun

		__syncthreads();

		assert(FrontierSizeNode < MAX_TARGET_FRONTIER_SIZE);
		#if defined(CUDA_SIMILARITY_DEBUG)
			if (threadIdx.x == 0 && blockIdx.x == 5) {
				printf("Level %d \t Frontier Size: %d\n", level, FrontierSizeNode);
				for (int i = 0; i < FrontierSizeNode; i++)
					printf("%d ", (int) FrontierNode[i]);
				printf("\n");
			}
			__syncthreads();
		#endif
	}

    int logSize = logValue<BLOCKDIM_SIM_TARGET, MIN_VW_SIM_TARGET, MAX_VW_SIM_TARGET> (FrontierSizeNode);
	#define fun(a)	NodeFrontier_to_Score <(a)> (devNode, devEdge, FrontierSizeNode, devLabel, devDegree,\
                                    devQueryBFS, devBFSSIM, pitchQueryBFS, pitchBFSSIM, query_nodes);

	def_SWITCH(logSize);
    #undef fun
}


template <int VW_SIZE>
__device__ __forceinline__ void NodeFrontierExpansion (	const t_edge_t* __restrict__ devNode,
													const t_node_t* __restrict__ devEdge,
													mark_t* __restrict__ devMarked,
													int& FrontierSizeNode) {

	t_node_t* FrontierNode = (t_node_t*) (SMem + FRONTIER_SIM_TARGET_POS);
	int NodeQueue[MAX_REG_ARRAY_SIZE];
	int nodeFounds = 0;

	for (int i = threadIdx.x / VW_SIZE; i < FrontierSizeNode; i += BLOCKDIM_SIM_TARGET / VW_SIZE) {
		assert(nodeFounds < MAX_REG_ARRAY_SIZE);

		const int actualNode = FrontierNode[i];
		const int end = devNode[ actualNode + 1 ];

		for (int j = devNode[ actualNode ] + (threadIdx.x % VW_SIZE); j < end; j += VW_SIZE) {
			const int dest = devEdge[ j ];
	#if CAS
			if (!atomicCAS(devMarked + dest, false, true)) {
	#else
			if (!devMarked[dest]) {
				devMarked[dest] = true;
	#endif
				NodeQueue[nodeFounds++] = dest;
			}
		}
	}
	int* Totale = (int*) SMem;
	int* TempNode = (int*) (SMem + TEMP_NODE_SIM_TARGET_POS);
	const int warpID = WarpID();

	int n = nodeFounds;
	WarpExclusiveScan<>::Add(n, TempNode + warpID);

	__syncthreads();

    const int N_OF_WARP = BLOCKDIM_SIM_TARGET / WARP_SIZE;
	if (threadIdx.x < N_OF_WARP) {
		int sum = TempNode[threadIdx.x];
        int total;
        WarpExclusiveScan<N_OF_WARP>::Add(sum, total);
        if (LaneID() == N_OF_WARP - 1)
		      Totale[0] = total + FrontierSizeNode;
		TempNode[threadIdx.x] = sum + FrontierSizeNode;
	}
	__syncthreads();

	FrontierSizeNode = Totale[0];
    assert(FrontierSizeNode < MAX_TARGET_FRONTIER_SIZE);

	n += TempNode[warpID];
	for (int i = 0; i < nodeFounds; i++)
		FrontierNode[n + i] = NodeQueue[i];
}


template <int VW_SIZE>
__device__ __forceinline__ void NodeFrontier_to_Score (	const t_edge_t* __restrict__ devNode,
											const t_node_t* __restrict__ devEdge,
											const int FrontierSizeNode,
											const label_t* __restrict__ devLabel,
											const t_node_t* __restrict__ devDegree,
											const bool* __restrict__ devQueryBFS,
											int* __restrict__ devBFSSIM,
											const int pitchQueryBFS,
											const int pitchBFSSIM,
											const int query_nodes) {


	int* TempScore = (int*) (SMem + TEMP_NODE_SIM_TARGET_POS);
	t_node_t* FrontierNode = (t_node_t*) (SMem + FRONTIER_SIM_TARGET_POS);

	const int warpID = WarpID();
    const int N_OF_VW = BLOCKDIM_SIM_TARGET / VW_SIZE;
    const int N_OF_WARP = BLOCKDIM_SIM_TARGET / WARP_SIZE;

if (LOW_MEMORY) {
	for (int Q = 0; Q < query_nodes; Q++) {
		int score = 0;
		for (int i = threadIdx.x / VW_SIZE; i < FrontierSizeNode; i += N_OF_VW) {
			const int actualNode = FrontierNode[i];
			const int end = devNode[ actualNode + 1 ];

			for (int j = devNode[ actualNode ] + (threadIdx.x % VW_SIZE); j < end; j += VW_SIZE) {
				const int dest = devEdge[ j ];
				const int sourceLabel = devLabel[actualNode];
				const int destLabel = devLabel[dest];
				if (devQueryBFS[Q * pitchQueryBFS + sourceLabel * MAX_LABELS + destLabel])
					score += devDegree[dest];
			}
		}

	    WarpReduce<>::Add(score, TempScore + warpID);

		__syncthreads();

		if (threadIdx.x < N_OF_WARP) {
			int sum = TempScore[threadIdx.x];
            WarpReduce<N_OF_WARP>::Add(sum, devBFSSIM + Q * pitchBFSSIM + blockIdx.x);
		}
	}
}
else
{
    int* Totale = (int*) SMem;
	t_node_t2 EdgeQueue[MAX_REG_ARRAY_SIZE];
	int edgeFounds = 0;

	for (int i = threadIdx.x / VW_SIZE; i < FrontierSizeNode; i += N_OF_VW) {
		assert(edgeFounds < MAX_REG_ARRAY_SIZE);

		const int actualNode = FrontierNode[i];

		const int end = devNode[ actualNode + 1 ];
		for (int j = devNode[ actualNode ] + (threadIdx.x % VW_SIZE); j < end; j += VW_SIZE) {
			const int dest = devEdge[ j ];
			EdgeQueue[edgeFounds].x = actualNode;
			EdgeQueue[edgeFounds++].y = dest;
		}
	}

	int n = edgeFounds;
    WarpExclusiveScan<>::Add(n, TempScore + warpID);

	__syncthreads();

	if (threadIdx.x < N_OF_WARP) {
		int sum = TempScore[threadIdx.x];
        WarpExclusiveScan<N_OF_WARP>::Add(sum, Totale);
		TempScore[threadIdx.x] = sum;
	}

	__syncthreads();

	t_node_t2* FrontierEdge = (t_node_t2*) ( SMem + FRONTIER_SIM_TARGET_POS );
	int FrontierSizeEdge = Totale[0];

    assert(FrontierSizeEdge * 2 < MAX_TARGET_FRONTIER2_SIZE);

	n += TempScore[warpID];
	for (int i = 0; i < edgeFounds; i++)
		FrontierEdge[n + i] = EdgeQueue[i];

	__syncthreads();

	for (int K = 0; K < query_nodes; K++) {
		int score = 0;
		for (int i = threadIdx.x; i < FrontierSizeEdge; i += BLOCKDIM_SIM_TARGET) {
			const int source = FrontierEdge[i].x;
			const int dest = FrontierEdge[i].y;
			const int sourceLabel = devLabel[source];
			const int destLabel = devLabel[dest];
			if (devQueryBFS[K * pitchQueryBFS + sourceLabel * MAX_LABELS + destLabel])
				score += devDegree[dest];
		}

        WarpReduce<>::Add(score, TempScore + warpID);

		__syncthreads();

		if (threadIdx.x < N_OF_WARP) {
			int sum = TempScore[threadIdx.x];
            WarpReduce<N_OF_WARP>::Add(sum, devBFSSIM + K * pitchBFSSIM + blockIdx.x);
		}
	}
}
}

} //@device
} //@appagato
