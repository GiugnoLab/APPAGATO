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
#include <random>
#include <chrono>
#include "Host/approx_subgraph_iso.hpp"

namespace appagato {
namespace host {
/*
void approx_subgraph_iso::computeProbability(const int* Similarity_matrix, float* Probability) {
    for (int i = 0; i < Query.V; i++) {
        for (int j = 0; j < Target.V; j++) {
            const float degree_sim = Query.Degrees[i] == Target.Degrees[j] ? 1.0f :
                                     1.0f - static_cast<float>(std::abs(static_cast<int>(Target.Degrees[j])
                                                                        - static_cast<int>(Query.Degrees[i]))) /
                                     std::max(static_cast<int>(Target.Degrees[j]), static_cast<int>(Query.Degrees[i]));

            const float label_sim = Target.Labels[j] == Query.Labels[i] ? 1.0f : 0.005f;

            Probability[i * Target.V + j] = degree_sim * label_sim * Similarity_matrix[i * Target.V + j];
        }
    }
}*/


void approx_subgraph_iso::computeSeeds( const float* const Probability_prefixsum,
                                        const float* const Random,
                                        int* const QuerySeeds, int* const TargetSeeds) {


    for (int K = 0; K < NOF_SOLUTIONS; K++) {
        for (int i = 0; i < Query.V; i++) {
            for (int j = 1; j < Target.V; j++) {
                const float pred = Probability_prefixsum[i * Target.V + j - 1] - Random[K];
                const float value = Probability_prefixsum[i * Target.V + j] - Random[K];

                if (pred <= 0 && value > 0) {
                    QuerySeeds[K] = i;
                    TargetSeeds[K] = j;
                    goto L1;
                }
            }
        }
        QuerySeeds[K] = Query.V - 1;
        TargetSeeds[K] = Target.V - 1;
L1:     ;
    }
}

void approx_subgraph_iso::seed(const int* const Similarity_matrix, int* const QuerySeeds, int* const TargetSeeds) {
    float* Probability = new float[Target.V * Query.V];
    if (LabelSimilarityMatrix == NULL)
        computeProbability<false>(Similarity_matrix, Probability);
    else
        computeProbability<true>(Similarity_matrix, Probability);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution (0.0f, 1.0f);

    float* Probability_prefixsum = new float[Target.V * Query.V];
    std::partial_sum(Probability, Probability + Query.V * Target.V, Probability_prefixsum);

    float* Random = new float[NOF_SOLUTIONS];
    const float sum = Probability_prefixsum[Target.V * Query.V - 1];
    for (int i = 0; i < NOF_SOLUTIONS; i++)
        Random[i] = distribution(generator) * sum;

    computeSeeds(Probability_prefixsum, Random, QuerySeeds, TargetSeeds);

    delete[] Random;
    delete[] Probability;
    delete[] Probability_prefixsum;
}

} //@host
} //@appagato
