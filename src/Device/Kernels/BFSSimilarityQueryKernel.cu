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
#include <assert.h>
#include "Device/approx_subgraph_iso.cuh"
#include "Device/devUtil.cuh"
#include "XLib.hpp"
using namespace primitives;

#include "../config.h"

namespace appagato {
namespace device {

//------------------------------------------------------------------------------

template <int VW_SIZE>
__device__ __forceinline__ void warpVisitQuery (	const q_edge_t* __restrict__ devNode,
													const q_node_t* __restrict__ devEdge,
													int& FrontierSizeNode,
													int& FrontierSizeEdge,
													q_node_t* __restrict__ FrontierNode,
													q_node_t2* __restrict__ FrontierEdge,
													bool* __restrict__ MarkedQuery);


const int REMAIN_SMEM_ITEMS_QUERY = (SMem_Per_Block<char, BLOCKDIM_SIM_QUERY>::value -              // total shared memory
                                        (MAX_QUERY_SIZE * sizeof(bool)                              // BFS marked nodes
                                        + MAX_QUERY_SIZE * sizeof(q_node_t)                         // BFS Queue
                                        + ((BLOCKDIM_SIM_QUERY / WARP_SIZE) * 2 + 2) * sizeof(int)) // Two Reduction
                                    ) / sizeof(q_node_t2);                                          // source - dest array

//------------------------------------------------------------------------------

__global__ void bfsSimilarityQueryKernel (	const q_edge_t* __restrict__ devNode,
											const q_node_t* __restrict__ devEdge,
											const label_t* __restrict__ devLabel,
											bool* __restrict__ devQueryBFS,
											const int pitchBFS,
											const int query_nodes) {

	__shared__ q_node_t2 FrontierEdge[ REMAIN_SMEM_ITEMS_QUERY ];
	__shared__ q_node_t FrontierNode[ MAX_QUERY_SIZE ];
	if (threadIdx.x == 0)
		FrontierNode[0] = blockIdx.x;

	__shared__ bool MarkedQuery[ MAX_QUERY_SIZE ];
	for (int i = threadIdx.x; i < query_nodes; i += BLOCKDIM_SIM_QUERY)
		MarkedQuery[i] = i == blockIdx.x ? true : false;

	__syncthreads();

	int FrontierSizeNode = 1;
	int FrontierSizeEdge = 0;

	for (int level = 0; FrontierSizeNode > 0 && level < BFS_SIMILARITY_DEEP; ++level) {

		const int logSize = logValue<BLOCKDIM_SIM_QUERY, MIN_VW_SIM_QUERY, MAX_VW_SIM_QUERY>
                                    (FrontierSizeNode);

        #define fun(a)	warpVisitQuery <(a)> (devNode, devEdge, FrontierSizeNode,        \
                                            FrontierSizeEdge, FrontierNode, FrontierEdge, MarkedQuery);

		def_SWITCH(logSize);

        #undef fun
	}
	__syncthreads();

	for (int i = threadIdx.x; i < FrontierSizeEdge; i += BLOCKDIM_SIM_QUERY) {
		const int sourceLabel = devLabel[FrontierEdge[i].x];
		const int destLabel = devLabel[FrontierEdge[i].y];
		devQueryBFS[ blockIdx.x * pitchBFS + sourceLabel * MAX_LABELS + destLabel ] = true;
	}
}



//------------------------------------------------------------------------------

template <int VW_SIZE>
__device__ __forceinline__ void warpVisitQuery (	const q_edge_t* __restrict__ devNode,
													const q_node_t* __restrict__ devEdge,
													int& FrontierSizeNode,
													int& FrontierSizeEdge,
													q_node_t* __restrict__ FrontierNode,
													q_node_t2* __restrict__ FrontierEdge,
													bool* __restrict__ MarkedQuery) {

    const int N_OF_WARPS = BLOCKDIM_SIM_QUERY / WARP_SIZE;

	int NodeQueue[MAX_REG_ARRAY_SIZE / 2];
	int nodeFounds = 0;

	q_node_t2 EdgeQueue[MAX_REG_ARRAY_SIZE / 2];
	int edgeFounds = 0;

	for (int i = threadIdx.x / VW_SIZE; i < FrontierSizeNode; i += N_OF_WARPS) {
		assert(nodeFounds < MAX_REG_ARRAY_SIZE / 2 && edgeFounds < MAX_REG_ARRAY_SIZE / 2);

		const int actualNode = FrontierNode[i];
		const int end = devNode[ actualNode + 1 ];

		for (int j = devNode[ actualNode ] + (threadIdx.x % VW_SIZE); j < end; j += VW_SIZE) {
			const int dest = devEdge[ j ];

			if (!MarkedQuery[dest]) {
				MarkedQuery[dest] = true;
				NodeQueue[nodeFounds++] = dest;
			}

			EdgeQueue[edgeFounds].x = actualNode;
			EdgeQueue[edgeFounds++].y = dest;
		}
	}

	__shared__ int TotalNodes;
	__shared__ int TotalEdges;
	__shared__ int TempNode[N_OF_WARPS];
	__shared__ int TempEdge[N_OF_WARPS];
	const int warpID = WarpID();

	int n = nodeFounds;
	int n2 = edgeFounds;

	WarpExclusiveScan<>::Add(n, TempNode + warpID);
	WarpExclusiveScan<>::Add(n2, TempEdge + warpID);

	__syncthreads();

	if (threadIdx.x < N_OF_WARPS) {
		int sum = TempNode[threadIdx.x];
        WarpExclusiveScan<N_OF_WARPS>::Add(sum, &TotalNodes);
		TempNode[threadIdx.x] = sum;
	} else if (threadIdx.x >= 32 && threadIdx.x < 32 + N_OF_WARPS) {
		int sum = TempEdge[LaneID()];
        int total;
        WarpExclusiveScan<N_OF_WARPS>::Add(sum, total);
        if (LaneID() == N_OF_WARPS - 1)
		      TotalEdges = total + FrontierSizeEdge;
		TempEdge[LaneID()] = sum + FrontierSizeEdge;
	}

	__syncthreads();

	FrontierSizeNode = TotalNodes;
	FrontierSizeEdge = TotalEdges;

	n += TempNode[warpID];
	for (int i = 0; i < nodeFounds; i++)
		FrontierNode[n + i] = NodeQueue[i];

	n2 += TempEdge[warpID];
	for (int i = 0; i < edgeFounds; i++)
		FrontierEdge[n2 + i] = EdgeQueue[i];

    __syncthreads();
    /*if (blockIdx.x == 13 && threadIdx.x == 0) {
        for (int i = 0; i < TotalEdges; i++)
            printf("(%d,%d)\n", (int) FrontierEdge[i].x, (int) FrontierEdge[i].y);
        printf("\n");
    }*/
}

} //@device
} //@appagato
