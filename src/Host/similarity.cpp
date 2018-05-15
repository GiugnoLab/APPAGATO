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

namespace appagato {
namespace host {

    void approx_subgraph_iso::similarity(int* Similarity_matrix) {
        bool* Marked_matrix_query = new bool[MAX_LABELS * MAX_LABELS * Query.V]();
        q_node_t* Distance = new q_node_t[Target.V];
    	std::fill(Distance, Distance + Target.V, INF_QNODE);

        fast_queue::Queue<q_node_t> Queue_Q(Query.V);
        fast_queue::Queue<q_node_t> QueueTMP_Q(Query.V);
        for (int source = 0; source < Query.V; source++)
            BFS_Similarity<false>(Query, Marked_matrix_query, (q_node_t) source,
                                  Distance, Queue_Q, QueueTMP_Q);

        fast_queue::Queue<t_node_t> Queue_T(Target.V);
        fast_queue::Queue<t_node_t> QueueTMP_T(Target.V);
        for (int source = 0; source < Target.V; source++)
            BFS_Similarity<true>(Target, Marked_matrix_query, (t_node_t) source, Distance,
                                Queue_T, QueueTMP_T, Similarity_matrix);

        delete[] Marked_matrix_query;
        delete[] Distance;
    }
} //@host
} //@appagato
