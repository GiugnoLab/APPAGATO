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

#include "Kernels/ExtendKernel.cu"

namespace appagato {
namespace device {

void approx_subgraph_iso::extend(const int* devQuerySeeds, const int* devTargetSeeds) {
	dim3 dimBlock(16, 8);
	dim3 dimGrid( Div(Target.V, dimBlock.x), Div(NOF_SOLUTIONS, dimBlock.y) );
	cuda_util::fill<<< dimGrid, dimBlock >>> (devTargetMarked, NOF_SOLUTIONS, Target.V, INF_QNODE, pitchMarked);

    cuda_util::fill<<< Div(NOF_SOLUTIONS, 128) , 128 >>>(devValidSolution, NOF_SOLUTIONS, true);
    __ENABLE(CHECK_ERROR, __CUDA_ERROR("Extend Init"))
    __ENABLE(PRINT_INFO, __PRINT("-> Extend Kernel" <<
                                "\t\tExtend Kernel Max Target Frontier: " << MAX_TARGET_EXT_FRONTIER_SIZE << std::endl);)

    if (LabelSimilarityMatrix == NULL) {
    	bfsKernel_Matching<false> <<< NOF_SOLUTIONS, BLOCKDIM_EXT, SMem_Per_Block<char, BLOCKDIM_EXT>::value >>>
    						(	devQuerySeeds, devTargetSeeds,
    							devNodeQuery, devEdgeQuery,
                                devLabelQuery, devNodeTarget, devEdgeTarget, devLabelTarget,
    							devQueryMatch, pitchQueryMatch / sizeof(q_node_t),
                                devTargetMatch, pitchTargetMatch / sizeof(t_node_t),
    							devRandomState, devValidSolution,
    							devTargetMarked, pitchMarked / sizeof(q_node_t), Query.V,
                                Target.V, devLabelSimilarityMatrix);
    } else {
        bfsKernel_Matching<true> <<< NOF_SOLUTIONS, BLOCKDIM_EXT, SMem_Per_Block<char, BLOCKDIM_EXT>::value >>>
                    (	devQuerySeeds, devTargetSeeds,
                        devNodeQuery, devEdgeQuery,
                        devLabelQuery, devNodeTarget, devEdgeTarget, devLabelTarget,
                        devQueryMatch, pitchQueryMatch / sizeof(q_node_t),
                        devTargetMatch, pitchTargetMatch / sizeof(t_node_t),
                        devRandomState, devValidSolution,
                        devTargetMarked, pitchMarked / sizeof(q_node_t), Query.V,
                        Target.V, devLabelSimilarityMatrix);
    }

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("Extend Kernel"))

	cudaMemcpy2DAsync(QueryMatch, Query.V * sizeof(q_node_t), devQueryMatch,
                      pitchQueryMatch, Query.V * sizeof (q_node_t),
                      NOF_SOLUTIONS, cudaMemcpyDeviceToHost);
	cudaMemcpy2DAsync(TargetMatch, Query.V * sizeof(t_node_t), devTargetMatch,
                      pitchTargetMatch, Query.V * sizeof (t_node_t),
                      NOF_SOLUTIONS, cudaMemcpyDeviceToHost);
	cudaMemcpy(ValidSolutions, devValidSolution, NOF_SOLUTIONS * sizeof (bool),
                      cudaMemcpyDeviceToHost);

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("Extend Copy Result"))
}

} //@device
} //@appagato
