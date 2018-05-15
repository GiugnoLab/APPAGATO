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
#pragma once
#include "../../config.h"
#include <string>
#include "XLib.hpp"
#include "types.hpp"

namespace graph {
    enum class G_TYPE {QUERY, TARGET};

    using LabelMap_t = typename fUtil::UniqueMap<std::string, label_t>;

    void readHeader(const char* File, int& V, int& E);

    template<typename node_t, typename edge_t>
	class GraphAPP {
		public:
            int V, E;
			edge_t *Nodes;
			node_t *Edges, *Degrees;
            label_t *Labels;

			GraphAPP(const node_t _V, const edge_t _E);
            ~GraphAPP();

            template<G_TYPE _G_TYPE>
			void read( const char* File, fUtil::UniqueMap<std::string, label_t>* LabelMap);
            void readSimilarityMatrix(const char* const File);
            bool isWeaklyConnected();
            void maxFrontiers();

			//void toBinary( const char* File );
	};
}

#include "GraphAPP.i.hpp"
