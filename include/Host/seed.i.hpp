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
namespace host {

template<bool LABEL_MATRIX>
void approx_subgraph_iso::computeProbability(const int* Similarity_matrix, float* Probability) {
    const float* LabelSimilarityMatrixTMP = LabelSimilarityMatrix;
    for (int i = 0; i < Query.V; i++) {
        q_node_t q_degree = Query.Degrees[i];
        label_t q_label = Query.Labels[i];
        for (int j = 0; j < Target.V; j++) {
            const float degree_sim = q_degree == Target.Degrees[j] ? 1.0f :
                                     1.0f - static_cast<float>(std::abs(static_cast<int>(Target.Degrees[j])
                                                                        - static_cast<int>(q_degree))) /
                                     std::max(static_cast<int>(Target.Degrees[j]), static_cast<int>(q_degree));

            const float label_sim = LABEL_MATRIX ? *LabelSimilarityMatrixTMP++ : (Target.Labels[j] == q_label ? 1.0f : 0.005f);

            *Probability++ = degree_sim * label_sim * (*Similarity_matrix++);
        }
    }
}

} //@host
} //@appagato
