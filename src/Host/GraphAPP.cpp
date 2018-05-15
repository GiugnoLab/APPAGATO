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
#include <fstream>
#include "Host/GraphAPP.hpp"
#include "XLib.hpp"

namespace graph {

void readHeader(const char* const File, int& V, int& E) {
    std::ifstream fin(File);
	fileUtil::checkRegularFile(fin);

	std::string GraphName;
	fin >> GraphName >> V;
    if (V <= 0)
    if (V <= 0)
        __ERROR("N. of Nodes <= 0 : " << File);
	fileUtil::skipLines(fin, V + 1);
	fin >> E;
	fin.close();
    if (E <= 0)
        __ERROR("N. of Edges <= 0 : " << File);

	#if (PRINT_INFO)
            StreamModifier::thousandSep();
    		std::cout << std::endl << "Read Header:\t" << fileUtil::extractFileName(std::string(File))
				  << std::endl << "       Name:\t" << "\"" << GraphName.substr(1)
                  << "\"\t\tNodes: " << V << "\tEdges: " << E << "\tDirected Edges: " << E * 2
			      << "\tSize: " << std::setprecision(1) << std::fixed << (float) fileUtil::fileSize(File) / (1<<20)
                  << " MB" << std::endl << std::endl;
            StreamModifier::resetSep();
	#endif
    E *= 2;
}

}   //@graph
