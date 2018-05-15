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
#include "XLib.hpp"
#include <numeric>

namespace graph {

template<typename node_t, typename edge_t>
GraphAPP<node_t, edge_t>::GraphAPP(const node_t _V, const edge_t _E) : V(_V), E(_E) {
    Nodes = new edge_t[V + 1];
    Edges = new node_t[E];
    Degrees = new node_t[V]();
    Labels = new label_t[V];
}

template<typename node_t, typename edge_t>
GraphAPP<node_t, edge_t>::~GraphAPP() {
    delete[] Nodes;
    delete[] Edges;
    delete[] Degrees;
    delete[] Labels;
}

template<typename node_t, typename edge_t>
template<G_TYPE _G_TYPE>
void GraphAPP<node_t, edge_t>::read( const char* const File, LabelMap_t* LabelMap ) {
    __ENABLE(PRINT_INFO, __PRINT("Reading Graph File..." << std::flush))

    std::ifstream fin(File);
	fileUtil::skipLines(fin, 2);

	for (int i = 0; i < V; i++) {
		std::string str;
		fin >> str;
        fileUtil::skipLines(fin);
		//Labels[i] = _G_TYPE == G_TYPE::QUERY ? LabelMap->find(str)->second : LabelMap->insertValue(str);
        Labels[i] = LabelMap->insertValue(str);
	}
    int n_of_lines;
    fin >> n_of_lines;
    auto TmpEdge = new node_t[n_of_lines][2];
    edge_t* TmpDegree = new edge_t[V]();

    for (int i = 0; i < n_of_lines; i++) {
        int index1, index2;
        fin >> index1 >> index2;

        TmpEdge[i][0] = index1;
        TmpEdge[i][1] = index2;
        TmpDegree[index1]++;
        TmpDegree[index2]++;
    }
    fin.close();
    Nodes[0] = 0;
    std::partial_sum(TmpDegree, TmpDegree + V, Nodes + 1);

    for (int i = 0; i < n_of_lines; i++) {
        const edge_t source = TmpEdge[i][0];
        const edge_t dest = TmpEdge[i][1];
        Edges[ Nodes[source] + Degrees[source]++ ] = dest;
        Edges[ Nodes[dest] + Degrees[dest]++ ] = source;
    }
    delete [] TmpEdge;
    delete [] TmpDegree;

    __ENABLE(PRINT_INFO, __PRINT("Complete!" << std::endl))
    #if (DEBUG_READ)
    if (V < 100) {
        printExt::host::printArray(Nodes, V + 1, "Nodes\n");
        printExt::host::printArray(Degrees, V, "Degrees\n");
        printExt::host::printArray(Edges, E, "Edges\n");
    }
    #endif
}

template<typename node_t, typename edge_t>
bool GraphAPP<node_t, edge_t>::isWeaklyConnected() {
    fast_queue::Queue<node_t> Queue(V);
    Queue.insert(0);

	std::vector<bool> BitMask(V, false);
	BitMask[0] = true;
	int count = 1;
	while (Queue.size() > 0) {
		const node_t actualNode = Queue.extract();
		for (int i = Nodes[actualNode]; i < Nodes[actualNode + 1]; ++i) {
			const edge_t dest = Edges[i];
			if (!BitMask[dest]) {
				BitMask[dest] = true;
				Queue.insert(dest);
				++count;
			}
		}
	}
	return count != V;
}


template<typename node_t, typename edge_t>
void GraphAPP<node_t, edge_t>::maxFrontiers() {
    fast_queue::Queue<node_t> Queue(V);
	node_t* Distance = new node_t[V];
	int max_visited_node = 0, max_visited_edge = 0;

	for (int source = 0; source < V; ++source) {
		std::fill(Distance, Distance + V, std::numeric_limits<node_t>::max() - 1);

		Distance[source] = 0;
        Queue.insert(source);
		int count_edge = Degrees[source];
		int count_node = 1;

		while (Queue.size() > 0) {
			count_node++;
			const node_t actualNode = Queue.extract();
			if (Distance[actualNode] == BFS_SIMILARITY_DEEP) break;
			for (int i = static_cast<int>(Nodes[actualNode]); i < static_cast<int>(Nodes[actualNode + 1]); ++i) {
				const node_t dest = Edges[i];
				if (Distance[dest] == std::numeric_limits<node_t>::max() - 1) {
					Distance[dest] = Distance[actualNode] + 1;
                    Queue.insert(dest);
					count_edge += Degrees[dest];
				}
			}
		}
		if (count_edge > max_visited_edge)
			max_visited_edge = count_edge;
		if (count_node > max_visited_node)
			max_visited_node = count_node;
	}
	std::cout 	<< "      BFSSim_DEEP: " << BFS_SIMILARITY_DEEP << std::endl
				<< "max_visited_edges: " << max_visited_edge << std::endl
				<< "max_visited_nodes: " << max_visited_node << std::endl
				<< "       max_degree: " << *std::max_element(Degrees, Degrees + V) << std::endl << std::endl;
}

} //@graph
