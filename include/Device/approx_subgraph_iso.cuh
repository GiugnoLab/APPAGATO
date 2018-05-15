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
#include "types.hpp"
#include "Host/approx_subgraph_iso.hpp"
#include <curand_kernel.h>
#include "XLib.hpp"

namespace appagato {
namespace device {

class approx_subgraph_iso : public appagato::host::approx_subgraph_iso {
    protected:
        void similarity(int* devSimilarity_matrix) override;
        void seed(const int* devSimilarity_matrix, int* devQuerySeeds, int* devTargetSeeds) override;
        void extend(const int* devQuerySeeds, const int* devTargetSeeds) override;

    private:
        q_edge_t* devNodeQuery;
        q_node_t* devEdgeQuery;
        label_t* devLabelQuery;
        q_node_t* devDegreeQuery;

        t_edge_t* devNodeTarget;
        t_node_t* devEdgeTarget;
        label_t* devLabelTarget;
        t_node_t* devDegreeTarget;
        curandState* devRandomState;

        float* devLabelSimilarityMatrix;

        // ---------------- SIMILARITY -----------------------------------------
        size_t pitchBFSSIM;
        size_t pitchQueryBFS;
        size_t pitchMarkedBFS;

        mark_t* devMarked_matrix_target;
        bool* devMarked_matrix_query;
        int* devSimilarity_matrix;

        void check_query_similarity(const bool* devMarked_matrix_query);
        void check_target_similarity(const bool* devMarked_matrix_query,
                                     const int* devSimilarity_matrix);

        // ------------------- SEED --------------------------------------------
        int* devQuerySeeds, *devTargetSeeds;
        float* devProbability, *devRandomDebug;

        void checkProbs(const int* devSimilarity_matrix, const float* devProbability,
                        float*& Probability_DEVICE);
        void checkPrefixScanProbs(const float* Probability_DEVICE, const float* devProbability);
        void checkSeed(const float* devRandomDebug, const float* devProbability,
                            const int* devQuerySeeds, const int* devTargetSeeds);

        // ---------------- EXTEND -----------------------------------------

        q_node_t* devTargetMarked;
        q_node_t* devQueryMatch;
        t_node_t* devTargetMatch;
        bool* devValidSolution;
        size_t pitchMarked;
        size_t pitchQueryMatch;
        size_t pitchTargetMatch;

    public:
        approx_subgraph_iso(graph::GraphAPP<q_node_t, q_edge_t>& Query,
                            graph::GraphAPP<t_node_t, t_edge_t>& Target,
                            const int NOF_SOLUTIONS);

        ~approx_subgraph_iso();

        void resolve() override;
};

__global__ void bfsSimilarityQueryKernel (	const q_edge_t* __restrict__ devNode,
											const q_node_t* __restrict__ devEdge,
											const label_t* __restrict__ devLabel,
											bool* __restrict__ devQueryBFS,
											const int pitchBFS,
											const int query_nodes);

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
											const int                    query_nodes);

__global__ void SeedCorrelationKernel(	const label_t* __restrict__  devLabelQuery,
										const label_t* __restrict__  devLabelTarget,
										const q_node_t* __restrict__ devDegreeQuery,
										const t_node_t* __restrict__ devDegreeTarget,
										const int* __restrict__      devMatrixSimilarity,
										float* __restrict__          devProbability,
										const int                    query_nodes,
										const int                    target_nodes,
										const int                    pitch);

__global__ void SeedSelectionKernel(const float* __restrict__   devProbability,
    								const int                   query_nodes,
    								const int                   target_nodes,
    								curandState* __restrict__   devRandomState,
    								float* __restrict__         devRandomDebug,
    								const int                   NOF_SOLUTIONS,
    								int* __restrict__           devQuerySeeds,
    								int* __restrict__           devTargetSeeds);

__global__ void bfsKernel_Matching (const int* __restrict__ devQuerySeeds,
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
									const int nof_queryNode);


} //@device
} //@appagato
