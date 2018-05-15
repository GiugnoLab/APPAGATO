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
#include "readGraph.h"

// -------------- PRIVATE HEADER -------------------------------

void readGraphPGSolver(	std::ifstream& fin, GraphWeight& graph, const int nof_lines, int& PlayerTH );

// -------------- IMPLEMENTATION -------------------------------

namespace readGraph {

void readMPG( const char* File, GraphWeight& graph, const int nof_lines) {
	std::cout << "Reading Graph File..." << std::flush;

	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	int PlayerTH;
	readGraphPGSolver(fin, graph, nof_lines, PlayerTH);

	fin.close();
	std::cout << "\tComplete!\n" << std::flush;

	graph.ToCSR();
	graph.setPlayerTH(PlayerTH);
}

}	//end namespace


void readGraphPGSolver(	std::ifstream& fin, GraphWeight& graph, const int nof_lines, int& PlayerTH) {
	initProgress(nof_lines);

	readUtil::skipLines(fin);
	fin >> PlayerTH;
	readUtil::skipLines(fin, 2);

	for (int lines = 0; lines < nof_lines; ++lines) {
		int index1, index2, weight;
		fin >> index1 >> index2 >> weight;

		graph.COO_Edges[lines] = make_int3(index1, index2, weight);

		readProgress(lines + 1);
		readUtil::skipLines(fin);
	}
	graph.COOSize = nof_lines;
}
