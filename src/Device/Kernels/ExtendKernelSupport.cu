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
#include "XLib.hpp"
using namespace primitives;


    const int DEBUG_TARGET_NODES = 535;
    const int DEBUG_TARGET_EDGES = 13554;
    const int DEBUG_QUERY_NODES = 23;
    const int DEBUG_QUERY_EDGES = 164;
/*const int          DEBUG_TARGET_NODES = std::numeric_limits<t_node_t>::max();
const unsigned int DEBUG_TARGET_EDGES = std::numeric_limits<t_edge_t>::max();
const int           DEBUG_QUERY_NODES = std::numeric_limits<q_node_t>::max();
const int           DEBUG_QUERY_EDGES = std::numeric_limits<q_edge_t>::max();*/

//----------------------------
// PARALLEL QUERY VISIT (WARP)
//----------------------------

__device__ __forceinline__ int warpVisitQuery(	const __restrict__ q_edge_t* devQueryNode,
												const __restrict__ q_node_t* devQueryEdge,
												const int queryMatch,
												int& QueryFrontierSize,
												curandState& localState,
												q_node_t* __restrict__ MarkedQueryMatch,
												q_node_t* __restrict__ QueryFrontier) {

	q_node_t Queues[MAX_REG_ARRAY_SIZE];
	int founds = 0;

    assert(queryMatch < DEBUG_QUERY_NODES);
	const int end = devQueryNode[ queryMatch + 1 ];
	for (int k = devQueryNode[ queryMatch ] + LaneID(); k < end; k += WARP_SIZE) {
		assert(founds < MAX_REG_ARRAY_SIZE);

		const int dest = devQueryEdge[ k ];
        assert(dest < DEBUG_QUERY_EDGES);

		if (MarkedQueryMatch[dest] == INF_QNODE) {
			Queues[founds++] = dest;
			MarkedQueryMatch[dest] = IN_QUEUE;
		}
	}
	int n = founds;
    int total;
    WarpExclusiveScan<>::AddBcast(n, total);
    assert(total + QueryFrontierSize < MAX_QUERY_SIZE);

	n += QueryFrontierSize;
	for (int i = 0; i < founds; i++)
		QueryFrontier[n + i] = Queues[i];

	QueryFrontierSize += total;

	q_node_t newMatch;

    /*#if defined(CUDA_EXTEND_DEBUG)
    if (blockIdx.x == DEBUG_BLOCK) {
        printf("threadIdx %d \t founds %d\n", threadIdx.x, founds);
    }
    #endif*/

	if (threadIdx.x == 0) {
#if defined(CUDA_EXTEND_DEBUG)
		curandState localStateB = localState;
#endif

		const int random = (int) (curand_uniform(&localState) * QueryFrontierSize); //- FLT_MIN
		newMatch = QueryFrontier[random];

#if defined(CUDA_EXTEND_DEBUG)
		if (blockIdx.x == DEBUG_BLOCK) {
			printf("Query Frontier  (%d):\n", QueryFrontierSize);
			for (int i = 0; i < QueryFrontierSize; i++)
				printf(" %d", (int) QueryFrontier[i]);
			printf("\n\tnew Match:\t%d\trandom: %d  (%f)\n", newMatch, random, curand_uniform(&localStateB));
		}
#endif
		QueryFrontier[random] = QueryFrontier[QueryFrontierSize - 1];
	}

	newMatch = __shfl(newMatch, 0);
	QueryFrontierSize--;

	return newMatch;
}

//---------------------------------
// PARALLEL QUERY BUILD MAPS (WARP)
//---------------------------------

__device__ void warpQueryBuildMaps(	const __restrict__ q_edge_t* devQueryNode,
									const __restrict__ q_node_t* devQueryEdge,
									const __restrict__ label_t* devLabelQuery,
									const int nof_matched,
									const int newQueryMatch,
									const q_node_t* MarkedQueryMatch) {
	int founds = 0;
	unsigned localBitMask[MAX_QUERY_SIZE/32];		// MAX QUERY SIZE : 256 (NODES)

	const int nCopy = _Div(nof_matched, 32);
    assert(nCopy < MAX_QUERY_SIZE/32);
	for (int i = 0; i < nCopy; i++)
		localBitMask[i] = 0;

    assert(newQueryMatch < DEBUG_QUERY_NODES);
	const int end = devQueryNode[ newQueryMatch + 1 ];
	for (int k = devQueryNode[ newQueryMatch ] + LaneID(); k < end; k += WARP_SIZE) {
		const int dest = devQueryEdge[ k ];
        assert(dest < DEBUG_QUERY_EDGES);

		const q_node_t pos = MarkedQueryMatch[dest];
		if (pos < IN_QUEUE) {
			localBitMask[pos >> 5] |= ( 1 << (pos & 31) );
			founds++;
			//cuPrintf("BuildMaps -- threadIdx.x %d \t dest %d \t mask %X \t\n", threadIdx.x, dest, localBitMask[0]);
		}
	}

	#pragma UNROLL 5
	for (int i = 16; i >= 1; i >>= 1) {					// FUSIONE BITMASK
		for (int j = 0; j < nCopy; j++)
			localBitMask[j] |= __shfl_xor((int) localBitMask[j], i);

		founds += __shfl_xor(founds, i);
	}

	unsigned* QueryFeatures = (unsigned*) (SMem + QUERY_FEATURES_POS);
	for (int i = LaneID(); i < nCopy; i += 32)
		QueryFeatures[i] = localBitMask[i];

    if (LaneID() == 0) {
    	short2* nof_features = (short2*) (SMem + N_FEATURES_POS);
    	nof_features[0] = make_short2(devLabelQuery[newQueryMatch], founds);
    }
#if defined(CUDA_EXTEND_DEBUG)
	if (threadIdx.x == 0 && blockIdx.x == DEBUG_BLOCK) {
		printf("\tFeatures:\t");
		for (int i = 0; i < nof_matched; i++)
			printf("%d ", (bool) (QueryFeatures[i >> 5] & ( 1 << (i & 31) )));
		printf("\t(%d)\n", founds);
		//cuPrintf("\t%d   %d \n", nof_features[0].x, nof_features[0].y);
	}
#endif
}

//-----------------------------
// PARALLEL TARGET VISIT (WARP)
//-----------------------------

__device__ void warpVisitTarget(	const t_edge_t* __restrict__ devTargetNode,
									const t_node_t* __restrict__ devTargetEdge,
									q_node_t* devTargetMarked,
									const int targetMatch,
									const int TargetFrontierSize) {

	int Queues[MAX_REG_ARRAY_SIZE];
	int founds = 0;

	const int end = devTargetNode[ targetMatch + 1 ];
	for (int k = devTargetNode[ targetMatch ] + LaneID(); k < end; k += WARP_SIZE) {
		assert(founds < MAX_REG_ARRAY_SIZE);//if (k >= 32)
        assert(k < DEBUG_TARGET_EDGES);

		const int dest = devTargetEdge[ k ];
        assert(dest < DEBUG_TARGET_NODES);

		if (devTargetMarked[dest] == INF_QNODE) {
			Queues[founds++] = dest;
			devTargetMarked[dest] = IN_QUEUE;
		}
	}
	int n = founds;
    int total;
    WarpExclusiveScan<>::AddBcast(n, total);
	assert(total + TargetFrontierSize < MAX_TARGET_EXT_FRONTIER_SIZE);

	n += TargetFrontierSize;

	t_node_t* TargetFrontier = (t_node_t*) ( SMem + TARGET_FRONTIER_POS );
	for (int i = 0; i < founds; i++) {
		TargetFrontier[ n + i ] = Queues[i];
        assert(Queues[i] < DEBUG_TARGET_EDGES);
    }

    if (LaneID() == 0) {
    	int* TotaleSH = (int*) &SMem[ TOTAL_POS ];
    	TotaleSH[0] = TargetFrontierSize + total;
    }
}


template<int VW_SIZE>
__device__ int ParallelProbs(const int TargetFrontierSize, float random,
                            float QueueProbs[NOF_PROBS_REGISTER], const int pFounds);

//------------------------------
// PARALLEL TARGET PROBABILITIES
//------------------------------

template<bool LABEL_MATRIX, int VW_SIZE>
__device__ int warpVisitProbsTarget(	const __restrict__ t_edge_t* devTargetNode,
										const __restrict__ t_node_t* devTargetEdge,
										const __restrict__ label_t* devLabelTarget,
										q_node_t* devTargetMarked,
										const int TargetFrontierSize,
										const int nof_matched,
										curandState& localStateB,
                                        const int pitchQueryMatch,
                                        float* devLabelSimilarityMatrix,
                                        const int nof_targetNode,
                                        q_node_t* __restrict__ devQueryMatch) {

	unsigned* QueryFeatures = (unsigned*) (SMem + QUERY_FEATURES_POS);
	t_node_t* TargetFrontier = (t_node_t*) (SMem + TARGET_FRONTIER_POS);

	float* TargetProbs = (float*) (SMem + TARGET_PROBS_POS);	// |TargetProbs| max 32
	bool inRegister = TargetFrontierSize > 32;
    q_node_t query_match = devQueryMatch[blockIdx.x * pitchQueryMatch + nof_matched];

	float QueueProbs[NOF_PROBS_REGISTER];
	int pFounds = 0;

	short2* nof_features = (short2*) (SMem + N_FEATURES_POS);
	const int QueryLabel = nof_features[0].x;
	const int local_nof_features = nof_features[0].y;

	unsigned localQueryFeatures[8];
	const int nCopy = _Div(nof_matched, 32);
	for (int i = 0; i < nCopy; i++)
		localQueryFeatures[i] = QueryFeatures[i];

	float totalWarp = 0;
	const int maxLoop = (TargetFrontierSize + BLOCKDIM_EXT / VW_SIZE - 1) >> (LOG2<BLOCKDIM_EXT>::value - LOG2<VW_SIZE>::value);
	for (int t = threadIdx.x / VW_SIZE, p = 0; p < maxLoop; t += BLOCKDIM_EXT / VW_SIZE, p++) {

		int founds = 0;
		if (t < TargetFrontierSize) {
			const int index = TargetFrontier[ t ];
            assert(index < DEBUG_TARGET_NODES);
			const int end = devTargetNode[ index + 1 ];
			for (int k = devTargetNode[ index ] + (threadIdx.x % VW_SIZE); k < end; k += VW_SIZE) {
                assert(k < DEBUG_TARGET_EDGES);
				const int dest = devTargetEdge[ k ];
                assert(dest < DEBUG_TARGET_NODES);

				const int pos = devTargetMarked[ dest ];
				if (pos < IN_QUEUE && (localQueryFeatures[pos >> 5] & (1 << (pos & 31)))) {
					founds++;
					//cuPrintf("CheckMaps -- threadIdx.x %d \t dest %d \t mask %X \t\n", threadIdx.x, dest, localQueryFeatures[0]);
				}
			}

            WarpReduce<MIN<VW_SIZE, WARP_SIZE>::value>::AddBcast(founds);

			const int TScore = founds;
			const float NodeProbs = LABEL_MATRIX ? devLabelSimilarityMatrix[query_match *nof_targetNode + index] :
                                    (QueryLabel == devLabelTarget[index] ?
                                    __fdividef(TScore, local_nof_features) :
                                    __fdividef(TScore, local_nof_features) * 0.005);
			totalWarp += NodeProbs;

			if (inRegister) {
				if ((threadIdx.x % VW_SIZE) == (p & MOD2<VW_SIZE>::value))
					QueueProbs[pFounds++] = NodeProbs;
			} else
				TargetProbs[t] = NodeProbs;
#if defined(CUDA_EXTEND_DEBUG)
			if (blockIdx.x == DEBUG_BLOCK && (threadIdx.x % VW_SIZE) == 0)
				printf("Index %d \t Vwarp %d \t Node %d :: Score %d \t label %d \t local_nof_features %d \t probs %f\n",
					t, threadIdx.x / VW_SIZE, TargetFrontier[ t ], TScore, QueryLabel == devLabelTarget[index], local_nof_features, NodeProbs);
#endif
		}
	}

	//----------------------------
	// ******** FASE 2 **********
	//----------------------------
//------------------------------ TargetFrontierSize <= 32 ----------------------

	const float random = curand_uniform(&localStateB);
	if (TargetFrontierSize <= 32) {
		__syncthreads();
		if (threadIdx.x < 32) {
			float prob = threadIdx.x < TargetFrontierSize ? TargetProbs[threadIdx.x] : 0.0f;
            WarpInclusiveScan<>::Add(prob);

            const float totalProbs = __shfl(prob, 31);
#if defined(CUDA_EXTEND_DEBUG)
			if (blockIdx.x == DEBUG_BLOCK && threadIdx.x == 0)
				printf("\nSUM of Probability: %f \t random_position: %f \t threads_values:\n", totalProbs, totalProbs == 0 ? random * TargetFrontierSize : random * totalProbs);
#endif
			if (totalProbs == 0)
				return threadIdx.x == 0 ? (int) (random * TargetFrontierSize) : -1;

			const float value = prob - random * totalProbs;
			const float pred = __shfl_up(value, 1);
#if defined(CUDA_EXTEND_DEBUG)
			if (blockIdx.x == DEBUG_BLOCK && threadIdx.x < TargetFrontierSize)
				printf("\tTid %d \t prob %f \t value %f \t pred %f\n", threadIdx.x, prob, value, pred);
#endif
			if ((threadIdx.x == 0 || pred <= 0) && value > 0)
				return threadIdx.x;
		}
		return -1;
//------------------------------------------------------------------------------
	}
	if (VW_SIZE <= 2)
		totalWarp += __shfl_xor(totalWarp, 2, 4);
	if (VW_SIZE <= 4)
		totalWarp += __shfl_xor(totalWarp, 4, 8);
	if (VW_SIZE <= 8)
		totalWarp += __shfl_xor(totalWarp, 8, 16);
	if (VW_SIZE <= 16)
		totalWarp += __shfl_xor(totalWarp, 16, 32);

	float* Temp = (float*) (SMem + TEMP_POS);
	float* TotaleSH = (float*) (SMem + TOTAL_POS);

	Temp[WarpID()] = totalWarp;
	__syncthreads();
    const int N_OF_WARPS = BLOCKDIM_EXT/WARP_SIZE;

	if (threadIdx.x < N_OF_WARPS) {
		float sum = Temp[threadIdx.x];
        WarpReduce<N_OF_WARPS>::Add(sum, TotaleSH);
	}

	__syncthreads();
	const float totalProbs = TotaleSH[0];

#if defined(CUDA_EXTEND_DEBUG)
	if (threadIdx.x == 0 && blockIdx.x == DEBUG_BLOCK)
		printf("\t totalProbs: %f \t randomPOS: %f\n", totalProbs, totalProbs == 0 ? random * TargetFrontierSize : random * totalProbs);
#endif
	if (totalProbs == 0)
		return threadIdx.x == 0 ? (int) (random * TargetFrontierSize) : -1;

	return ParallelProbs<VW_SIZE>(TargetFrontierSize, random * totalProbs, QueueProbs, pFounds);
}

				//t * BLOCKDIM_E + (BLOCKDIM_E/VW_SIZE) * (threadIdx.x & WARP_MOD) + VwarpId;
#define		INDEX		(((t - 1) << LOG2<BLOCKDIM_EXT>::value) + ((threadIdx.x & MOD2<VW_SIZE>::value) << (LOG2<BLOCKDIM_EXT>::value - LOG2<VW_SIZE>::value)) + (threadIdx.x >> LOG2<VW_SIZE>::value))


//-----------------------------
// PARALLEL GETBY_PROBABILITIES
//-----------------------------

template<int VW_SIZE>
__device__ int ParallelProbs(const int TargetFrontierSize, const float randomTT, float QueueProbs[NOF_PROBS_REGISTER], const int pFounds) {
	float2* Temp = (float2*) (SMem + TEMP_POS);
	float* totale = (float*) (SMem + TOTAL_POS);
    const int N_OF_WARPS = BLOCKDIM_EXT/WARP_SIZE;

	const int warpId = WarpID();
	Temp[warpId] = make_float2(0, 0);
	float prob = 0;
	float localTotale = 0;

const int maxLoop = _Div(TargetFrontierSize, BLOCKDIM_EXT);
int t;
for (t = 0; t < maxLoop && localTotale < randomTT; t++) {
	prob = t < pFounds ? QueueProbs[t] : 0;

    WarpInclusiveScan<>::Add(prob);
    if (LaneID() == 31)
	   Temp[warpId].x = prob;

	__syncthreads();
	if (threadIdx.x < N_OF_WARPS) {
        float sum = Temp[threadIdx.x].x;
        WarpInclusiveScan<N_OF_WARPS>::Add(sum, totale);
		Temp[threadIdx.x].x = sum + localTotale;
	}
	__syncthreads();

	localTotale += totale[0];
}
	prob += Temp[warpId].x;

	const float value = prob - randomTT;
	float pred = __shfl_up(value, 1);

	if (__all(value <= 0))
		Temp[warpId] = make_float2(0, 0);
	else if (value > 0 && (LaneID() == 0 || pred <= 0)) {
		Temp[warpId] = make_float2(INDEX, value);
//#if defined(CUDA_EXTEND_DEBUG)
	//printf("threadIdx.x %d \t %d \t %f\n",  threadIdx.x, INDEX, value);
//#endif
	}
	__syncthreads();

	if (threadIdx.x < BLOCKDIM_EXT/WARP_SIZE) {
		const float2 value2 = Temp[threadIdx.x];
		pred = __shfl_up(value2.y, 1);

		//assert(!__all(value2.y <= 0));
		if ((threadIdx.x == 0 || pred <= 0) && value2.y > 0)
			return (int) value2.x;
	}
	return -1;
}
