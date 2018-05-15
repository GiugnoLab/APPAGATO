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
#include "types.hpp"
#include "Host/GraphAPP.hpp"

namespace appagato {
namespace host {

class approx_subgraph_iso {
    protected:
        const graph::GraphAPP<q_node_t, q_edge_t>& Query;
        const graph::GraphAPP<t_node_t, t_edge_t>& Target;
        const int NOF_SOLUTIONS;
        q_node_t* QueryMatch;
    	t_node_t* TargetMatch;
        bool* ValidSolutions;

        virtual void similarity(int* similarity_matrix);
        virtual void seed(const int* const similarity_matrix, int* const QuerySeeds, int* const TargetSeeds);
        virtual void extend(const int* const QuerySeeds, const int* const TargetSeeds);

        template<bool LABEL_MATRIX>
        void extendSupport(const int* const QuerySeeds, const int* const TargetSeeds);

        template< bool TARGET, class node_t, class edge_t>
        void BFS_Similarity(const graph::GraphAPP<node_t, edge_t>& Graph,
                            bool* Marked_matrix_query,
                            const node_t source, q_node_t* Distance,
                            fast_queue::Queue<node_t>& Queue,
                            fast_queue::Queue<node_t>& QueueTMP,
                            int* BFS_SIM = NULL);

        template<bool LABEL_MATRIX>
        void computeProbability(const int* Similarity_matrix, float* Probability);

        void computeSeeds( const float* const Probability_prefixsum,
                           const float* const Random,
                           int* const QuerySeeds, int* const TargetSeeds);


    private:
        template<class node_t, class edge_t>
        inline bool isEdge(edge_t* Node, node_t* Edge, node_t u, node_t v);
        inline int GetByProb(const float* const probs, const int N, float random);

    public:
        approx_subgraph_iso(const graph::GraphAPP<q_node_t, q_edge_t>& _Query,
                            const graph::GraphAPP<t_node_t, t_edge_t>& _Target,
                            const int _NOF_SOLUTIONS);
        ~approx_subgraph_iso();
        virtual void resolve();
        virtual void printSolutions() final;
        virtual void calculateCost() final;
        virtual void uniqueSolutions() final;

        static void readSimilarityMatrix(const char* const File,
                                         const int query_nodes,
                                         const int target_nodes);
        static float* LabelSimilarityMatrix;
};
} //@host
} //@appagato

#include "approx_subgraph_iso.i.hpp"
#include "similarity.i.hpp"
#include "seed.i.hpp"
#include "extend.i.hpp"
