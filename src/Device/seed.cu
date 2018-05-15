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
using namespace numeric;
using namespace timer_cuda;

#include "Kernels/SeedKernel.cu"

#include "cub/cub.cuh"

namespace appagato {
namespace device {

void approx_subgraph_iso::seed(const int* devSimilarity_matrix, int* devQuerySeeds, int* devTargetSeeds) {
    const int    BLOCKDIM_X_PROB  =	 16;
    const int    BLOCKDIM_Y_PROB  =	 16;
    const int      BLOCKDIM_SEED  =  128;

    cuda_util::fill <<< Div(NOF_SOLUTIONS,256), 256 >>>(devQuerySeeds, NOF_SOLUTIONS, Query.V - 1);
    cuda_util::fill <<< Div(NOF_SOLUTIONS,256), 256 >>>(devTargetSeeds, NOF_SOLUTIONS, Target.V - 1);
    __ENABLE(CHECK_ERROR, cudaMalloc(&devRandomDebug     , NOF_SOLUTIONS * sizeof(float));)

    // ---------------------------  PROBABILITY --------------------------------

	__ENABLE(CHECK_ERROR, __CUDA_ERROR("Cuda Seed Init"))
    __ENABLE(PRINT_INFO, __PRINT("-> Correlation Kernel" << std::endl);)

    dim3 dimBlock(BLOCKDIM_X_PROB, BLOCKDIM_Y_PROB);
    dim3 dimGrid( Div(Target.V, BLOCKDIM_X_PROB), Div(Query.V, BLOCKDIM_Y_PROB));

    if (LabelSimilarityMatrix == NULL) {
    	SeedCorrelationKernel<false><<<dimGrid, dimBlock>>>
            (devLabelQuery, devLabelTarget, devDegreeQuery, devDegreeTarget,
            devSimilarity_matrix, devProbability, Query.V, Target.V, pitchBFSSIM / sizeof (int), NULL);
    } else {
        SeedCorrelationKernel<true><<<dimGrid, dimBlock>>>
            (devLabelQuery, devLabelTarget, devDegreeQuery, devDegreeTarget,
            devSimilarity_matrix, devProbability, Query.V, Target.V, pitchBFSSIM / sizeof (int), devLabelSimilarityMatrix);
    }
    __ENABLE(CHECK_ERROR, __CUDA_ERROR("Seed Kernel"))
    __attribute__((unused)) float* Probability_DEVICE;
	__ENABLE(CHECK_RESULT, checkProbs(devSimilarity_matrix, devProbability, Probability_DEVICE);)

	// --------------------------- PREFIX-SCAN ---------------------------------

    __ENABLE(PRINT_INFO, __PRINT("-> Probability Matrix Prefix-Scan" << std::endl);)

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, devProbability, devProbability, Query.V * Target.V);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, devProbability, devProbability, Query.V * Target.V);

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("Prefix Scan Kernel"))
	__ENABLE(CHECK_RESULT, checkPrefixScanProbs(Probability_DEVICE, devProbability);)

	// --------------------------- SEED ----------------------------------------

    __ENABLE(PRINT_INFO, __PRINT("-> Seed Selection Kernel" << std::endl);)

	dim3 dimGridSeed(Div(Target.V, BLOCKDIM_SEED), Query.V);

	SeedSelectionKernel <<< dimGridSeed, BLOCKDIM_SEED >>>
                            (devProbability, Query.V, Target.V, devRandomState,
							devRandomDebug, NOF_SOLUTIONS, devQuerySeeds, devTargetSeeds);

    __ENABLE(CHECK_ERROR, __CUDA_ERROR("Seed Kernel 2"))
	__ENABLE(CHECK_RESULT, checkSeed(devRandomDebug, devProbability, devQuerySeeds, devTargetSeeds);)
}


void approx_subgraph_iso::checkProbs(const int* devSimilarity_matrix, const float* devProbability,
                                     float*& Probability_DEVICE) {

    int* Similarity_matrix_DEVICE = new int[Query.V * Target.V];
    cudaMemcpy2D(Similarity_matrix_DEVICE, Target.V * sizeof (int),
            devSimilarity_matrix, pitchBFSSIM, Target.V * sizeof (int), Query.V,
            cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------

	float* Probability_HOST = new float[Target.V * Query.V];
    if (LabelSimilarityMatrix == NULL)
        computeProbability<false>(Similarity_matrix_DEVICE, Probability_HOST);
    else
        computeProbability<true>(Similarity_matrix_DEVICE, Probability_HOST);

    //--------------------------------------------------------------------------

    Probability_DEVICE = new float[Query.V * Target.V];
    cudaMemcpy(Probability_DEVICE, devProbability, Query.V * Target.V * sizeof (float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < Query.V * Target.V; ++i) {
		if ( std::isnan(Probability_DEVICE[i]) || std::isinf(Probability_DEVICE[i]) )
			__ERROR("Wrong Seed Probs NAN or INF");
	}
    //--------------------------------------------------------------------------

	if( !std::equal(Probability_HOST, Probability_HOST + Target.V * Query.V,
                    Probability_DEVICE, compareFloatABS_Str<std::micro>()) )
		__ERROR("Wrong Seed Probs Correlation");

	std::cout << "\t<> Seed Probs Correlation CORRECT" << std::endl << std::endl;
	delete Probability_HOST;
	delete Similarity_matrix_DEVICE;
}


void approx_subgraph_iso::checkPrefixScanProbs(const float* Probability_DEVICE, const float* devProbability) {
	float* Probs_prefixsum_HOST = new float[Target.V * Query.V];
	std::partial_sum(Probability_DEVICE, Probability_DEVICE + Query.V * Target.V, Probs_prefixsum_HOST);

	for (int i = 0; i < Query.V * Target.V; ++i) {
		if ( std::isnan(Probs_prefixsum_HOST[i]) || std::isinf(Probs_prefixsum_HOST[i]) )
			__ERROR("Wrong Seed Probs NAN or INF");
	}

    //--------------------------------------------------------------------------

    float* Probs_prefixsum_DEVICE = new float[Query.V * Target.V];
    cudaMemcpy(Probs_prefixsum_DEVICE, devProbability, Query.V * Target.V * sizeof (float), cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------

	if( !std::equal(Probs_prefixsum_HOST, Probs_prefixsum_HOST + Target.V * Query.V,
                    Probs_prefixsum_DEVICE, compareFloatRel_Str<std::milli>()) )
		__ERROR("Wrong Seed Prefix Probs");

	std::cout << "\t<> Seed Prefix Probs CORRECT" << std::endl << std::endl;
	delete Probs_prefixsum_HOST;
	delete Probability_DEVICE;
}



void approx_subgraph_iso::checkSeed(const float* devRandomDebug, const float* devProbability,
                                    const int* devQuerySeeds, const int* devTargetSeeds) {

	float* Random_DEVICE = new float[NOF_SOLUTIONS];
	cudaMemcpy(Random_DEVICE, devRandomDebug, NOF_SOLUTIONS * sizeof (float), cudaMemcpyDeviceToHost);

    float* Probs_prefixsum_DEVICE = new float[Query.V * Target.V];
    cudaMemcpy(Probs_prefixsum_DEVICE, devProbability, Query.V * Target.V * sizeof (float), cudaMemcpyDeviceToHost);

	int* QuerySeeds_HOST = new int[NOF_SOLUTIONS];
	int* TargetSeeds_HOST = new int[NOF_SOLUTIONS];

    computeSeeds(Probs_prefixsum_DEVICE, Random_DEVICE, QuerySeeds_HOST, TargetSeeds_HOST);

	int* QuerySeeds_DEVICE = new int[NOF_SOLUTIONS];
	int* TargetSeeds_DEVICE = new int[NOF_SOLUTIONS];
	cudaMemcpy(QuerySeeds_DEVICE, devQuerySeeds, NOF_SOLUTIONS * sizeof (int), cudaMemcpyDeviceToHost);
	cudaMemcpy(TargetSeeds_DEVICE, devTargetSeeds, NOF_SOLUTIONS * sizeof (int), cudaMemcpyDeviceToHost);

    /*for (int i = 0; i < NOF_SOLUTIONS; i++) {
        if (QuerySeeds_HOST[i] != QuerySeeds_DEVICE[i]) {
                std::cout << i << " -> " << QuerySeeds_HOST[i] << " "<< QuerySeeds_DEVICE[i] << std::endl;
                std::exit(1);
        }
    }*/

	if( !std::equal(QuerySeeds_HOST, QuerySeeds_HOST + NOF_SOLUTIONS, QuerySeeds_DEVICE) )
		__ERROR("Wrong Query Seed");
	if( !std::equal(TargetSeeds_HOST, TargetSeeds_HOST + NOF_SOLUTIONS, TargetSeeds_DEVICE) )
		__ERROR("Wrong Target Seed");

	std::cout << "\t<> Query/Target Seed CORRECT" << std::endl << std::endl;
    delete Random_DEVICE;
	delete QuerySeeds_HOST;
	delete TargetSeeds_HOST;
    delete QuerySeeds_DEVICE;
    delete TargetSeeds_DEVICE;
}

} //@device
} //@appagato
