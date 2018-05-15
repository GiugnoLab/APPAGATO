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
#include "Host/approx_subgraph_iso.hpp"
#include <algorithm>
#include <bitset>
#include <assert.h>
#include <random>
#include "XLib.hpp"

namespace appagato {
namespace host {

inline int approx_subgraph_iso::GetByProb(const float* const probs, const int N, float random) {
	for (int i = 0; i < N; i++) {
		random -= probs[i];
		if (random < 0.0f)
			return i;
	}
	return N - 1;
}

template<bool LABEL_MATRIX>
void approx_subgraph_iso::extendSupport(const int* const QuerySeeds, const int* const TargetSeeds) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution (0.0f, 1.0f);

    std::fill(ValidSolutions, ValidSolutions + NOF_SOLUTIONS, true);

    fast_queue::Queue<q_node_t> Queue_query(Query.V);
    fast_queue::Queue<t_node_t> Queue_target(Target.V);

	q_node_t* QueryMark = new q_node_t[Query.V];
    q_node_t* TargetMark = new q_node_t[Target.V];

	std::bitset<256> QueryFeature;
	float* Target_probability = new float[Target.V];

	for (int K = 0; K < NOF_SOLUTIONS; K++) {
		std::fill(QueryMark, QueryMark + Query.V, INF_QNODE);
		std::fill(TargetMark, TargetMark + Target.V, INF_QNODE);

        const q_node_t query_seed = QuerySeeds[K];
        QueryMatch[K * Query.V] = query_seed;
        QueryMark[query_seed] = 0;  //posizione zero

        const t_node_t target_seed = TargetSeeds[K];
        TargetMatch[K * Query.V] = target_seed;
        TargetMark[target_seed] = 0; //posizione zero

        // ------------ VISIT QUERY FROM LAST MATCH ----------------------------

		for (int i = Query.Nodes[query_seed]; i < Query.Nodes[query_seed + 1]; i++) {
			const q_node_t dest = Query.Edges[i];
			if (QueryMark[dest] == INF_QNODE) {
                Queue_query.insert(dest);
				QueryMark[dest] = IN_QUEUE;
			}
		}

        t_node_t last_target_match = target_seed;
		for (int pos = 1; pos < Query.V; pos++) {
			const int randomQueue = distribution(generator) * Queue_query.size();
            const int last_query_match = Queue_query.extract(randomQueue);
            assert(last_query_match >= 0 && last_query_match < Query.V);

			QueryMatch[K * Query.V + pos] = last_query_match;
			QueryMark[last_query_match] = pos;

			int q_score = 0;
			QueryFeature.reset();
			for (int i = Query.Nodes[last_query_match]; i < Query.Nodes[last_query_match + 1]; i++) {
				const q_node_t dest = Query.Edges[i];
				if (QueryMark[dest] == INF_QNODE) {
                    Queue_query.insert(dest);
					QueryMark[dest] = IN_QUEUE;
				} else if (QueryMark[dest] != IN_QUEUE) {
					QueryFeature[QueryMark[dest]] = true;
					q_score++;
				}
			}
            assert(q_score >= 1);

			// ------------ VISIT TARGET FROM LAST MATCH -----------------------

			for (int i = Target.Nodes[last_target_match]; i < static_cast<int>(Target.Nodes[last_target_match + 1]); i++) {
				const t_node_t dest = Target.Edges[i];
				if (TargetMark[dest] == INF_QNODE) {
                    Queue_target.insert(dest);
					TargetMark[dest] = IN_QUEUE;
				}
			}
			if (Queue_target.size() == 0) {
				ValidSolutions[K] = false;
				break;
			}

			for (int j = 0; j < Queue_target.size(); j++) {

				t_node_t node_in_queue = Queue_target.get(j);//TargetQueue[j];
				int t_score = 0;

				for (int i = Target.Nodes[node_in_queue]; i < (int) Target.Nodes[node_in_queue + 1]; i++) {
					const t_node_t dest = Target.Edges[i];
					if (TargetMark[dest] != IN_QUEUE && TargetMark[dest] != INF_QNODE && QueryFeature[TargetMark[dest]])
						t_score++;
				}
				Target_probability[j] = LABEL_MATRIX ? ((float) t_score / q_score) * LabelSimilarityMatrix[last_query_match * Target.V + node_in_queue] :
                                        (Target.Labels[node_in_queue] == Query.Labels[last_query_match] ?
                                        ((float) t_score / q_score) : ((float) t_score / q_score) * 0.005f);
			}
            const float sum = std::accumulate(Target_probability, Target_probability + Queue_target.size(), 0);
            //assert(sum > 0);
            //assert(false);
			const int last_target_match_pos = GetByProb(Target_probability, Queue_target.size(), distribution(generator) * sum);
            last_target_match = Queue_target.extract(last_target_match_pos);

			TargetMatch[K * Query.V + pos] = last_target_match;
			TargetMark[last_target_match] = last_target_match_pos;
		}
        Queue_query.reset();
        Queue_target.reset();
	}
	delete[] QueryMark;
	delete[] TargetMark;
	delete[] Target_probability;
}

} //@host
} //@appagato
