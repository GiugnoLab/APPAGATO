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
#include <curand_kernel.h>
#include "Device/approx_subgraph_iso.cuh"
#include "XLib.hpp"
using namespace numeric;
using namespace timer_cuda;

namespace appagato {
namespace device {

    __global__ void RandSetup_kernel(curandState* devRandomState, const unsigned seed, const int nof_state) {
    	const int ID = threadIdx.x + blockIdx.x * blockDim.x;
    	if (ID < nof_state)
    		curand_init(seed, ID, 0, &devRandomState[ID]);
    }

    void approx_subgraph_iso::resolve() {
        __ENABLE(PRINT_INFO, __PRINT("    --------=: DEVICE :=--------" << std::endl));
        #if defined(GET_TIME)
            Timer<DEVICE> TM(2, 22);//, StreamModifier::FG_L_RED
            TM.start();
        #endif

        __ENABLE(PRINT_INFO, __PRINT("  --------=: Similarity :=--------"
                                    "\t\tBFS Similarity Deep:\t" << BFS_SIMILARITY_DEEP << std::endl));

        similarity(devSimilarity_matrix);

        #if defined(GET_TIME)
            TM.getTime("Similarity");
            TM.start();
        #endif

        __ENABLE(PRINT_INFO, __PRINT("     --------=: Seed :=--------" << std::endl))
        seed(devSimilarity_matrix, devQuerySeeds, devTargetSeeds);

        #if defined(GET_TIME)
            TM.getTime("Seed");
            TM.start();
        #endif

        __ENABLE(PRINT_INFO, __PRINT("\t  --------=: Extend :=--------" << std::endl));
        extend(devQuerySeeds, devTargetSeeds);

        #if defined(GET_TIME)
            TM.getTime("Extend");
        #endif
    }

    approx_subgraph_iso::approx_subgraph_iso(
                    graph::GraphAPP<q_node_t, q_edge_t>& Query,
                    graph::GraphAPP<t_node_t, t_edge_t>& Target,
                    const int NOF_SOLUTIONS) :
                    appagato::host::approx_subgraph_iso(Query, Target, NOF_SOLUTIONS) {

        #if defined(CUDA_SIMILARITY_DEBUG) || defined(CUDA_EXTEND_DEBUG)
    		size_t limit;
    		cudaThreadGetLimit(&limit, cudaLimitPrintfFifoSize);
    		cudaThreadSetLimit(cudaLimitPrintfFifoSize, limit * 100);
    	#endif

        #if (PRINT_INFO) || (CHECK_ERROR)
            #if (PRINT_INFO)
                cuda_util::memInfoCUDA(
            #elif (CHECK_ERROR)
                cuda_util::memCheckCUDA(
            #endif
    	            NOF_SOLUTIONS * 2 * sizeof (curandState)    +	    // RANDOM init
                    (Query.V + 1) * sizeof(q_edge_t)            +		// Query	NODE
                    Query.E * sizeof(q_node_t)                  +		// Query	EDGE
					Query.V * sizeof(label_t)                   +		// Query	LABEL
					Query.V * sizeof(q_node_t)	               	+		// Query	DEGREE
					(Target.V + 1) * sizeof (t_edge_t)          +		// Target	NODE
					Target.E * sizeof (t_node_t)	            +		// Target	EDGE
					Target.V * sizeof (label_t)	                +		// Target	LABEL
					Target.V * sizeof (t_node_t)	            +		// Target	DEGREE
                    // SIMLARITY
                    Target.V * Query.V * sizeof(int)                 +       // Similarity Matrix
                    MAX_LABELS * MAX_LABELS * sizeof(bool) * Query.V +       // Marked Query
                    Target.V * Target.V *sizeof(mark_t)              +       // Marked Target
                    // SEED
                    NOF_SOLUTIONS * Query.V * sizeof(int)	         +        // Query    Seeds
					NOF_SOLUTIONS * Target.V * sizeof(int)           +	      // Target   Seeds
                    Query.V * Target.V * sizeof(float)			     +        // devProbability
                    //Extend
                    NOF_SOLUTIONS * Target.V * sizeof(q_node_t)      +        // TargetMarked
                    NOF_SOLUTIONS * Query.V * sizeof(q_node_t)       +        // Query Match
                    NOF_SOLUTIONS * Query.V * sizeof(t_node_t)       +        // Target Match
                    NOF_SOLUTIONS * sizeof(bool));                            // Valid Solutions
        #endif

    	cudaMalloc(&devRandomState , NOF_SOLUTIONS * 2 * sizeof (curandState));
    	cudaMalloc(&devNodeQuery   , (Query.V + 1) * sizeof (q_edge_t));
    	cudaMalloc(&devEdgeQuery   , Query.E * sizeof (q_node_t));
    	cudaMalloc(&devLabelQuery  , Query.V * sizeof (label_t));
    	cudaMalloc(&devDegreeQuery , Query.V * sizeof (q_node_t));

    	cudaMalloc(&devNodeTarget   , (Target.V + 1) * sizeof (t_edge_t));
    	cudaMalloc(&devEdgeTarget   , Target.E * sizeof (t_node_t));
    	cudaMalloc(&devLabelTarget  , Target.V * sizeof (label_t));
    	cudaMalloc(&devDegreeTarget , Target.V * sizeof (t_node_t));

        if (LabelSimilarityMatrix != NULL)
            cudaMalloc(&devLabelSimilarityMatrix , Query.V * Target.V * sizeof (float));

        //SIMILARITY
        cudaMallocPitch(&devSimilarity_matrix, &pitchBFSSIM, Target.V * sizeof(int), Query.V);
        cudaMallocPitch(&devMarked_matrix_query  , &pitchQueryBFS  , MAX_LABELS * MAX_LABELS * sizeof(bool) , Query.V);
        cudaMallocPitch(&devMarked_matrix_target , &pitchMarkedBFS , Target.V * sizeof(mark_t)              , Target.V);
        //SEED
        cudaMalloc(&devProbability     , Query.V * Target.V * sizeof(float) );
        cudaMalloc(&devQuerySeeds      , NOF_SOLUTIONS * sizeof(int));
        cudaMalloc(&devTargetSeeds     , NOF_SOLUTIONS * sizeof(int));
        //EXTEND
        cudaMallocPitch(&devTargetMarked, &pitchMarked     , Target.V * sizeof(q_node_t), NOF_SOLUTIONS);
    	cudaMallocPitch(&devQueryMatch  , &pitchQueryMatch , Query.V * sizeof(q_node_t), NOF_SOLUTIONS);
    	cudaMallocPitch(&devTargetMatch , &pitchTargetMatch, Query.V * sizeof(t_node_t), NOF_SOLUTIONS);
    	cudaMalloc(&devValidSolution    , NOF_SOLUTIONS * sizeof(bool));

    	__ENABLE(CHECK_ERROR, __CUDA_ERROR("Graph Allocation"));
    	__ENABLE(PRINT_INFO, __PRINT("-> Graph Allocation Completed" << std::endl));

        cudaMemcpyAsync(devNodeQuery    , Query.Nodes   , (Query.V + 1) * sizeof (q_edge_t) , cudaMemcpyHostToDevice);
    	cudaMemcpyAsync(devEdgeQuery    , Query.Edges   , Query.E * sizeof (q_node_t)       , cudaMemcpyHostToDevice);
    	cudaMemcpyAsync(devLabelQuery   , Query.Labels  , Query.V * sizeof (label_t)        , cudaMemcpyHostToDevice);
    	cudaMemcpyAsync(devDegreeQuery  , Query.Degrees , Query.V * sizeof (q_node_t)       , cudaMemcpyHostToDevice);

    	cudaMemcpyAsync(devNodeTarget   , Target.Nodes   , (Target.V + 1) * sizeof (t_edge_t) , cudaMemcpyHostToDevice);
    	cudaMemcpyAsync(devEdgeTarget   , Target.Edges   , Target.E * sizeof (t_node_t)       , cudaMemcpyHostToDevice);
    	cudaMemcpyAsync(devLabelTarget  , Target.Labels  , Target.V * sizeof (label_t)        , cudaMemcpyHostToDevice);
    	cudaMemcpyAsync(devDegreeTarget , Target.Degrees , Target.V * sizeof (t_node_t)       , cudaMemcpyHostToDevice);

        if (LabelSimilarityMatrix != NULL)
            cudaMemcpyAsync(devLabelSimilarityMatrix , LabelSimilarityMatrix, Query.V * Target.V * sizeof (float), cudaMemcpyHostToDevice);

        std::srand(std::time(NULL));
    	RandSetup_kernel<<<Div(NOF_SOLUTIONS * 2, 128), 128>>> (devRandomState, std::rand(), NOF_SOLUTIONS * 2);

        __ENABLE(CHECK_ERROR, __CUDA_ERROR("Graph Copy To Device"));
        __ENABLE(PRINT_INFO, __PRINT("-> Graph Copy Completed" << std::endl));
    }

    approx_subgraph_iso::~approx_subgraph_iso() {
        cudaFree(devRandomState);
    	cudaFree(devNodeQuery);
    	cudaFree(devEdgeQuery);
    	cudaFree(devLabelQuery);
    	cudaFree(devDegreeQuery);

    	cudaFree(devNodeTarget);
    	cudaFree(devEdgeTarget);
    	cudaFree(devLabelTarget);
    	cudaFree(devDegreeTarget);

        //SIMILARITY
        cudaFree(devSimilarity_matrix);
        cudaFree(devMarked_matrix_query);
        cudaFree(devMarked_matrix_target);
        //SEED
        cudaFree(devProbability);
        cudaFree(devQuerySeeds);
        cudaFree(devTargetSeeds);
        //EXTEND
        cudaFree(devTargetMarked);
    	cudaFree(devQueryMatch);
    	cudaFree(devTargetMatch);
    	cudaFree(devValidSolution);
    }
}
}
