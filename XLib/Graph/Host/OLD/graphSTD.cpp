/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

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
#include "graphSTD.h"

using node_t = int;

GraphSTD::GraphSTD(const int _V, const int _E, const graph::GDirection GraphDirection) :
				V(_V), E(_E), Direction(GraphDirection) {
	try {
		Queue.init(V);
		Visited.resize(V);
		COO_Edges = new int2[ E ];

		OutNodes = new int[V + 1];
		OutNodes2 = new int2[ V ];
		OutEdges = new node_t[ E ];
		OutDegree = new int[ V ]();
		if (GraphDirection == graph::UNDIRECTED)
			return;
		InNodes = new int[V + 1];
		InNodes2 = new int2[ V + 1 ];
		InEdges = new node_t[ E ];
		InDegree = new int[ V ]();
	}
	catch(std::bad_alloc& exc) {
  		error("OUT OF MEMORY: Graph too Large !!");
	}
}



void GraphSTD::ToCSR() {
	std::cout << "        COO To CSR...\t\t" << std::flush;

	for (int i = 0; i < COOSize; i++) {
		const node_t source = COO_Edges[i].x;
		const node_t dest = COO_Edges[i].y;
		OutDegree[source]++;
		if (Direction == graph::UNDIRECTED)
			OutDegree[dest]++;
		else if (Direction == graph::DIRECTED)
			InDegree[dest]++;
	}

	OutNodes[0] = 0;
	std::partial_sum(OutDegree, OutDegree + V, OutNodes + 1);
	for (int i = 0; i < V; i++)
		OutNodes2[i] = make_int2(OutNodes[i], OutNodes[i + 1]);

	int* TMP = new int[V]();
	for (int i = 0; i < COOSize; i++) {
		const node_t source = COO_Edges[i].x;
		const node_t dest = COO_Edges[i].y;
		OutEdges[ OutNodes[source] + TMP[source]++ ] = dest;
		if (Direction == Graph::UNDIRECTED)
			OutEdges[ OutNodes[dest] + TMP[dest]++ ] = source;
	}

	if (Direction == DIRECTED) {
		InNodes[0] = 0;
		std::partial_sum(InDegree, InDegree + V, InNodes + 1);
		for (int i = 0; i < V; i++)
			InNodes2[i] = make_int2(InNodes[i], InNodes[i + 1]);

		std::fill(TMP, TMP + V, 0);
		for (int i = 0; i < COOSize; ++i) {
			const node_t dest = COO_Edges[i].y;
			InEdges[ InNodes[dest] + TMP[dest]++ ] = COO_Edges[i].x;
		}
	}
	delete TMP;
	std::cout << "Complete!\n\n" << std::flush;
}


void  GraphSTD::Dimacs10ToCOO() {
	std::cout << " Dimacs10th to COO...\t\t" << std::flush;
	int count_Edges = 0;
	for (int i = 0; i < V; i++) {
		for (int j = OutNodes[i]; j < OutNodes[i + 1]; j++) {
			const node_t dest = OutEdges[j];
			COO_Edges[count_Edges++] = make_int2(i, dest);
		}
	}
	this->COOSize = count_Edges;
	std::cout << "Complete!\n\n" << std::flush;
}


void GraphSTD::print() {
	printExt::printArray(COO_Edges, COOSize, "COO Edges\n");
	printExt::printArray(OutNodes, V + 1, "OutNodes\t");
	printExt::printArray(OutEdges, E, "OutEdges\t");
	printExt::printArray(OutDegree, V, "OutDegree\t");
	if (Direction == Graph::UNDIRECTED)
		return;
	printExt::printArray(InNodes, V + 1, "InNodes\t\t");
	printExt::printArray(InEdges, E, "InEdges\t\t");
	printExt::printArray(InDegree, V, "InDegree\t");
}


void GraphSTD::DegreeAnalisys() {
	StreamModifier::thousandSep();
	const float avg             = (float) E / V;
	const float stdDev          = fUtil::stdDeviation (OutDegree, V, avg);
	const int zeroDegree        = std::count (OutDegree, OutDegree + V, 0);
	const int oneDegree         = std::count (OutDegree, OutDegree + V, 1);
	std::pair<int*,int*> minmax = std::minmax_element (OutDegree, OutDegree + V);

	std::cout << std::setprecision(1)
			  << "          Avg:  " << avg    << "\t\tOutDegree 0:  " << std::left << std::setw(14) << zeroDegree << fUtil::perCent(zeroDegree, V) << " %" << std::endl
			  << "     Std. Dev:  " << stdDev << "\t\tOutDegree 1:  " << std::left << std::setw(14) << oneDegree << fUtil::perCent(oneDegree, V) << " %" << std::endl
			  << "          Min:  " << *minmax.first    << "\t\t" << std::endl
			  << "          Max:  " << *minmax.second   << "\t\t" << std::endl;
	if (Direction == Graph::DIRECTED)
		std::cout << "\t\t\t\t InDegree 0:  " << std::count (InDegree, InDegree + V, 0) << std::endl
				  << "\t\t\t\t InDegree 1:  " << std::count (InDegree, InDegree + V, 1) << std::endl;
	std::cout << std::endl;
	StreamModifier::resetSep();
}

#include "graphSTD_BFS.cpp"
