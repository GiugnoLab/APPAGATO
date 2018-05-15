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

namespace appagato {
namespace host {

template< bool TARGET, class node_t, class edge_t>
void approx_subgraph_iso::BFS_Similarity(	const graph::GraphAPP<node_t, edge_t>& Graph,
                    						bool* Marked_matrix_query,
                                            const node_t source, q_node_t* Distance,
                                            fast_queue::Queue<node_t>& Queue,
                                            fast_queue::Queue<node_t>& QueueTMP,
                                            int* BFS_SIM) {

	Queue.insert(source);
	QueueTMP.insert(source);
	Distance[source] = 0;

	while (!Queue.isEmpty()) {
	    const node_t actualNode = Queue.extract();
        assert(actualNode < Graph.V);

		if (Distance[actualNode] == BFS_SIMILARITY_DEEP) break;

		for (int i = static_cast<int>(Graph.Nodes[actualNode]); i < static_cast<int>(Graph.Nodes[actualNode + 1]); ++i) {
			const node_t dest = Graph.Edges[i];
            assert(dest < Graph.V);

			if (Distance[dest] == INF_QNODE) {
				Distance[dest] = Distance[actualNode] + 1;
                Queue.insert(dest);
                QueueTMP.insert(dest);
			}
			if (TARGET) {
				for (int j = 0; j < Query.V; j++) {
					if ( Marked_matrix_query[ j * MAX_LABELS * MAX_LABELS + Graph.Labels[actualNode] * MAX_LABELS + Graph.Labels[dest]] )
						BFS_SIM[j * Target.V + source] += Graph.Degrees[dest];
				}
			} else {
                //if (source == 13)
                //    std::cout << "(" << (int) actualNode << " " << (int) dest << ")     " << (int) Graph.Labels[actualNode] << " " << (int) Graph.Labels[dest] << std::endl;
				Marked_matrix_query[source * MAX_LABELS * MAX_LABELS + Graph.Labels[actualNode] * MAX_LABELS + Graph.Labels[dest]] = true;
            }
        }
	}

    for (int i = 0; i < QueueTMP.size(); ++i)
        Distance[QueueTMP.at(i)] = INF_QNODE;
    Queue.reset();
    QueueTMP.reset();
}

} //@host
} //@appagato
