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
#include "../include/GraphSTD.hpp"
#include <vector_types.h>	//int2
#include <sstream>
#include <iterator>
#include "XLib.hpp"
using namespace fUtil;

namespace graph {

namespace {
	void readGraphMatrixMarket(	std::ifstream& fin, GraphSTD& graph, const int nof_lines );
	void readGraphDimacs9	  (	std::ifstream& fin, GraphSTD& graph, const int nof_lines );
	void readGraphDimacs10	  (	std::ifstream& fin, GraphSTD& graph );
	void readGraphSnap		  (	std::ifstream& fin, GraphSTD& graph, const int nof_lines );
}

void GraphSTD::read( const char* File, const int nof_lines) {
	std::cout << "Reading Graph File..." << std::flush;

	std::ifstream fin(File);
	std::string s;
	fin >> s;
	fin.seekg(std::ios::beg);

	int dimacs10 = false;
	//MatrixMarket
	if (s.compare("%%MatrixMarket") == 0)
		readGraphMatrixMarket(fin, *this, nof_lines);
	//Dimacs10
	else if (s.compare("%") == 0 || fUtil::isDigit(s.c_str()) ) {
		readGraphDimacs10(fin, *this);
		dimacs10 = true;
	}
	//Dimacs9
	else if (s.compare("c") == 0 || s.compare("p") == 0)
		readGraphDimacs9(fin, *this, nof_lines);
	//SNAP
	else if (s.compare("#") == 0)
		readGraphSnap(fin, *this, nof_lines);

	fin.close();
	std::cout << "\tComplete!" << std::endl << std::flush;

	if (dimacs10 && Direction == EdgeType::UNDIRECTED )
		Direction = EdgeType::UNDEF_EDGE_TYPE;

	//ToCSR();

	if ( dimacs10 && Direction == EdgeType::UNDEF_EDGE_TYPE ) {
		//Dimacs10ToCOO();
		Direction = EdgeType::UNDIRECTED;
	}
}

namespace {

void readGraphMatrixMarket(	std::ifstream& fin, GraphSTD& graph, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	while (fin.peek() == '%')
		fileUtil::skipLines(fin);
	fileUtil::skipLines(fin);

	for (int lines = 0; lines < nof_lines; ++lines) {
		node_t index1, index2;
		fin >> index1 >> index2;
		index1--;
		index2--;

		//graph.COO_Edges[lines] = make_int2(index1, index2);

		progress.next(lines + 1);
		fileUtil::skipLines(fin);
	}
	//graph.COOSize = nof_lines;
}


void readGraphDimacs9(	std::ifstream& fin, const int nof_lines) {
	fUtil::Progress progress(nof_lines);

	char c;
	int lines = 0;
	std::string nil;
	while ((c = fin.peek()) != EOF) {
		if (c == 'a') {
			node_t index1, index2;
			fin >> nil >> index1 >> index2;
			index1--;
			index2--;

			//graph.COO_Edges[lines] = make_int2(index1, index2);

			lines++;
			progress.next(lines + 1);
		}
		fileUtil::skipLines(fin);
	}
	//graph.COOSize = lines;
}

void readGraphDimacs10(	std::ifstream& fin) {
	fUtil::initProgress(graph.V);
	while (fin.peek() == '%')
		fileUtil::skipLines(fin);
	fileUtil::skipLines(fin);

	int countEdges = 0;
	for (int lines = 0; lines < graph.V; lines++) {
		std::string str;
		std::getline(fin, str);

		std::istringstream stream(str);
		std::istream_iterator<std::string> iis(stream >> std::ws);

		int degree = std::distance(iis, std::istream_iterator<std::string>());

		std::istringstream stream2(str);
		for (int j = 0; j < degree; j++) {
			node_t index2;
			stream2 >> index2;

			//graph.COO_Edges[countEdges++] = make_int2(lines, index2 - 1);
		}
		fUtil::readProgress(lines + 1);
	}
	//graph.COOSize = countEdges;
}

void readGraphSnap(	std::ifstream& fin, const int nof_lines ) {
	fUtil::Progress progress(nof_lines);
	while (fin.peek() == '#')
		fileUtil::skipLines(fin);

	fUtil::UniqueMap<node_t> Map;
	for (int lines = 0; lines < nof_lines; lines++) {
		node_t ID1, ID2;
		fin >> ID1 >> ID2;
		node_t index1 = Map.insertValue(ID1);
		node_t index2 = Map.insertValue(ID2);

		//graph.COO_Edges[lines] = make_int2(index1, index2);

		fUtil::readProgress(lines + 1);
	}
//	graph.COOSize = nof_lines;
}

} //@anonimous
} //@graph
