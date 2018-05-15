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
#include <iostream>
#include <string>
#include "XLib.hpp"
using namespace timer;

int main(int argc, char** argv) {
    if (argc < 3)
        __ERROR("Wrong number of arguments");

    // ---------------------   copy LABELS -------------------------------------
    fileUtil::checkRegularFile(argv[1]);
    int V, E;
	std::ifstream fin(argv[1]);
    const int initPos = fin.tellg();

    std::string GraphName;
    fin >> GraphName >> V;
    if (V <= 0)
        throw std::runtime_error("Wrong number of vertices");

    fileUtil::skipLines(fin, V + 1);
    const int pos = fin.tellg();

    char* buffer = new char[pos - initPos + 1];
    fin.seekg(std::ios_base::beg);
    fin.read(buffer, pos);
    buffer[pos - initPos] = '\0';

    std::ofstream fout(argv[2]);
    fout << buffer;

    // ---------------------  edges header -------------------------------------
    fin >> E;
    if (E <= 0)
        throw std::runtime_error("Wrong number of edges");

    auto EdgesUndirected = new int[E][2];

    //Timer<HOST> TM;
    //TM.start();

    // ---------------------  edges reading ------------------------------------
    bool* Matrix = new bool[V * V]();

    int j = 0;
    for (int i = 0; i < E; ++i) {
		int index1, index2;
		fin >> index1 >> index2;

        bool* ptr = index1 > index2 ? &Matrix[(index1 * V) + index2] : &Matrix[(index2 * V) + index1];
        if (!*ptr) {
            EdgesUndirected[j][0] = index1;
            EdgesUndirected[j++][1] = index2;
            *ptr = true;
        }
	}

    // ---------------------  edges writing ------------------------------------

    //TM.getTime("Duplicate Removing");
    fout << j << std::endl;
    fin.close();
    for (int i = 0; i < j; i++)
        fout << EdgesUndirected[i][0] << " " << EdgesUndirected[i][1] << std::endl;
    fout.close();
}
