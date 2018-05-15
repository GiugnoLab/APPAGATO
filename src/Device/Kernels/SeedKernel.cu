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
namespace appagato {
namespace device {

template<bool LABEL_MATRIX>
__global__ void SeedCorrelationKernel(	const label_t* __restrict__  devLabelQuery,
										const label_t* __restrict__  devLabelTarget,
										const q_node_t* __restrict__ devDegreeQuery,
										const t_node_t* __restrict__ devDegreeTarget,
										const int* __restrict__      devSimilarityMatrix,
										float* __restrict__          devProbability,
										const int                    query_nodes,
										const int                    target_nodes,
										const int                    pitch,
                                        const float* __restrict__    devLabelSimilarityMatrix) {

	const int X = blockDim.x * blockIdx.x + threadIdx.x;
	const int Y = blockDim.y * blockIdx.y + threadIdx.y;

	if (X < target_nodes && Y < query_nodes) {
		const float degreeSim = devDegreeQuery[Y] == devDegreeTarget[X] ? 1.0f :
								1.0f - (float) abs(devDegreeQuery[Y] - devDegreeTarget[X])
                                / max(devDegreeQuery[Y], devDegreeTarget[X]);

		const float labelSim = LABEL_MATRIX ? devLabelSimilarityMatrix[Y * target_nodes + X] :
                               devLabelQuery[Y] == devLabelTarget[X] ? 1.0f : 0.005f;

		devProbability[Y * target_nodes + X] = degreeSim * labelSim * devSimilarityMatrix[Y * pitch + X];
    }
}


__global__ void SeedSelectionKernel(const float* __restrict__   devProbability,
    								const int                   query_nodes,
    								const int                   target_nodes,
    								curandState* __restrict__   devRandomState,
    								float* __restrict__         devRandomDebug,
    								const int                   NOF_SOLUTIONS,
    								int* __restrict__           devQuerySeeds,
    								int* __restrict__           devTargetSeeds) {

	curandState localState = devRandomState[ 0 ];
	const float total = devProbability[query_nodes * target_nodes - 1];

	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	float prob, predProb;
	if (X > 0 && X < target_nodes) {
		predProb = devProbability[blockIdx.y * target_nodes + X - 1];
		prob = devProbability[blockIdx.y * target_nodes + X];
	}

	for (int K = 0; K < NOF_SOLUTIONS; K++) {
		if (X > 0 && X < target_nodes) {
			const float random = curand_uniform(&localState) * total;
            #if (CHECK_ERROR)
    			if (X == 1 && blockIdx.y == 0)
    				devRandomDebug[K] = random;
            #endif
			const float pred = predProb - random;
			const float value = prob - random;
			if (pred <= 0 && value > 0) {
				devTargetSeeds[K] = X;
				devQuerySeeds[K] = blockIdx.y;
			}
		}
	}
}

} //@device
} //@appagato
