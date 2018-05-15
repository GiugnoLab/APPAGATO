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
#include "../../../config.h"
#include "XLib.hpp"
#include "Device/devUtil.cuh"

namespace appagato {
namespace device {

const int     QUERY_MARKED_POS = 0;                                                                       // 0
const int   QUERY_FRONTIER_POS = MAX_QUERY_SIZE * sizeof(q_node_t) + 1;  //+1 align                       // 254

const int       N_FEATURES_POS = QUERY_FRONTIER_POS + MAX_QUERY_SIZE * sizeof(q_node_t) + 1;  //+1 align  // 508
const int 	         TOTAL_POS = N_FEATURES_POS + sizeof(short2);       				                  // 512
const int 	          TEMP_POS = TOTAL_POS + sizeof(int) + 4;	//+4 align                                // 516
const int 	QUERY_FEATURES_POS = TEMP_POS + sizeof(float2) * 32;                                          // 772

const int     TARGET_PROBS_POS = TEMP_POS;
	// NOTA: TARGET_PROBS e TEMP sono disaccopiati. |TARGET_PROBS| = 4 * 32 < 8 * 32
const int  TARGET_FRONTIER_POS = QUERY_FEATURES_POS + 32;    // 8 = 256 (MAX_QUERY_SIZE) / 8

const int MAX_TARGET_EXT_FRONTIER_SIZE = (SMem_Per_Block<char, BLOCKDIM_EXT>::value - TARGET_FRONTIER_POS) / sizeof(t_node_t);

//#define CUDA_EXTEND_DEBUG
#define DEBUG_BLOCK 0

extern __shared__ char SMem[];
#include "ExtendKernelSupport.cu"

template<bool LABEL_MATRIX>
__global__ void bfsKernel_Matching (	const int* __restrict__ devQuerySeeds,
										const int* __restrict__ devTargetSeeds,
										const q_edge_t* __restrict__ devNodeQuery,
										const q_node_t* __restrict__ devEdgeQuery,
										const label_t* __restrict__ devLabelQuery,
										const t_edge_t* __restrict__ devNodeTarget,
										const t_node_t* __restrict__ devEdgeTarget,
										const label_t* __restrict__ devLabelTarget,

										q_node_t* __restrict__ devQueryMatch,
										const int pitchQueryMatch,
										t_node_t* __restrict__ devTargetMatch,
										const int pitchTargetMatch,

										curandState* __restrict__ devRandomState,
										bool* devValidSolution,

										q_node_t* __restrict__ devTargetMarked,
										const int pitch,
										const int nof_queryNode,
                                        const int nof_targetNode,
                                        float* devLabelSimilarityMatrix) {

	// FIRST QUERY MATCH  = QUERY SEED

	q_node_t queryMatch = (q_node_t) devQuerySeeds[ blockIdx.x ];
	q_node_t* MarkedQueryMatch = (q_node_t*) (SMem + QUERY_MARKED_POS);	// query marked stored in SMEM
	for (int i = threadIdx.x; i < nof_queryNode; i += BLOCKDIM_EXT)
		MarkedQueryMatch[i] = i != queryMatch ? INF_QNODE : 0;

	if (threadIdx.x == 0)
		devQueryMatch[ blockIdx.x * pitchQueryMatch ] = queryMatch;		// query matches stored in Global Memory

	// FIRST TARGET MATCH  = TARGET SEED

	devTargetMarked = devTargetMarked + blockIdx.x * pitch;

	t_node_t targetMatch = devTargetSeeds[ blockIdx.x ];
	if (threadIdx.x == 32) {
		devTargetMarked[targetMatch] = 0;
		devTargetMatch[ blockIdx.x * pitchTargetMatch ] = targetMatch;	// target matches stored in Global Memory
	}
	__syncthreads();

	// EXTEND
	q_node_t* QueryFrontier = (q_node_t*) (SMem + QUERY_FRONTIER_POS);	// query frontier salvati sulla SMEM

	curandState localStateQ = devRandomState[ blockIdx.x ];
	curandState localStateT = devRandomState[ blockIdx.x + gridDim.x];

	int QueryFrontierSize = 0;
	int TargetFrontierSize = 0;

for (int nof_matched = 1; nof_matched < nof_queryNode; nof_matched++) {

#if defined(CUDA_EXTEND_DEBUG)		// Query Matchs, Target Matchs
	__syncthreads();
	if (threadIdx.x == 0 && blockIdx.x == DEBUG_BLOCK) {
		printf("\nQuery Match:\n");
		for (int i = 0; i < nof_matched; i++)
			printf(" %d", (int) devQueryMatch[ blockIdx.x * pitchQueryMatch + i]);
		printf("\nTarget Match\n");
		for (int i = 0; i < nof_matched; i++)
			printf(" %d", (int) devTargetMatch[ blockIdx.x * pitchTargetMatch + i]);
		printf("\n");
        printf("\nTarget Frontier size:  %d\n", TargetFrontierSize);
	}
#endif
	// First WARP:
	//	1) Update Query Frontier
	//	2) Build Maps of Features
	if (threadIdx.x < 32) {
        assert(QueryFrontierSize >= 0);
        assert(queryMatch >= 0 && queryMatch < DEBUG_QUERY_NODES);

        queryMatch = warpVisitQuery(devNodeQuery, devEdgeQuery, queryMatch, QueryFrontierSize,
                                    localStateQ, MarkedQueryMatch, QueryFrontier);

        assert(queryMatch >= 0 && queryMatch < DEBUG_QUERY_NODES);

		if (threadIdx.x == 0)
			devQueryMatch[blockIdx.x * pitchQueryMatch + nof_matched] = queryMatch;
		MarkedQueryMatch[queryMatch] = (q_node_t) nof_matched;

		warpQueryBuildMaps(devNodeQuery, devEdgeQuery, devLabelQuery,
                            nof_matched, queryMatch, MarkedQueryMatch);
	}
	// Second WARP:
	//	1) Update Target Frontier
#if !defined(CUDA_EXTEND_DEBUG)
	else if (threadIdx.x < 64) {
#else
	__syncthreads(); if (threadIdx.x >= 32 && threadIdx.x < 64) {
#endif
        assert(targetMatch < DEBUG_TARGET_NODES);
        warpVisitTarget(devNodeTarget, devEdgeTarget, devTargetMarked, targetMatch, TargetFrontierSize);
	}
	__syncthreads();

	// Se la frontiera della Query o del target sono vuote allora termina (Valid = false)
	int* TotaleSH = (int*) (SMem + TOTAL_POS);
	TargetFrontierSize = TotaleSH[0];
	if (TargetFrontierSize == 0) {
		if (threadIdx.x == 0)
			devValidSolution[blockIdx.x] = false;
		return;
	}
	assert(TargetFrontierSize < NOF_PROBS_REGISTER * BLOCKDIM_EXT);

	int logSize = logValue<BLOCKDIM_EXT, MIN_VW_EXT, MAX_VW_EXT> (TargetFrontierSize);
	int newMatchPos;

    // Calcola le probabilità per ogni vertice in frontiera e scegli il nuovo Target Match
    #define fun(a)	newMatchPos = warpVisitProbsTarget <LABEL_MATRIX, (a)>\
                (devNodeTarget, devEdgeTarget, devLabelTarget, devTargetMarked, TargetFrontierSize,\
                nof_matched, localStateT, pitchQueryMatch, devLabelSimilarityMatrix, nof_targetNode, devQueryMatch);

	def_SWITCH(logSize);
    #undef fun

	t_node_t* TargetFrontier = (t_node_t*) (SMem + TARGET_FRONTIER_POS);

	if (newMatchPos >= 0) {    //only one
		// SWAP selected node with last
		t_node_t newTargetMatch = TargetFrontier[newMatchPos];
		TargetFrontier[newMatchPos] = TargetFrontier[TargetFrontierSize - 1];
        TargetFrontier[TargetFrontierSize - 1] = newTargetMatch;

		devTargetMarked[newTargetMatch] = nof_matched;
		devTargetMatch[blockIdx.x * pitchTargetMatch + nof_matched] = newTargetMatch;	// Insieme ??
	}
	__syncthreads();

    TargetFrontierSize--;
	targetMatch = TargetFrontier[TargetFrontierSize];

    __syncthreads();
}
}

} //@device
} //@appagato
