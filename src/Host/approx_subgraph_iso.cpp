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
#include <set>
#include <vector>
#include <iterator>
#include "Host/approx_subgraph_iso.hpp"
#include "XLib.hpp"
using namespace timer;

namespace appagato {
namespace host {

    approx_subgraph_iso::approx_subgraph_iso(const graph::GraphAPP<q_node_t, q_edge_t>& _Query,
                                             const graph::GraphAPP<t_node_t, t_edge_t>& _Target,
                                             const int _NOF_SOLUTIONS) :
                                             Query(_Query), Target(_Target), NOF_SOLUTIONS(_NOF_SOLUTIONS) {

        QueryMatch = new q_node_t[NOF_SOLUTIONS * Query.V];
        TargetMatch = new t_node_t[NOF_SOLUTIONS * Query.V];
        ValidSolutions = new bool[NOF_SOLUTIONS];
    }

    approx_subgraph_iso::~approx_subgraph_iso() {
        delete[] QueryMatch;
        delete[] TargetMatch;
        delete[] ValidSolutions;
    }


    void approx_subgraph_iso::resolve() {
        __ENABLE(PRINT_INFO, __PRINT("\t   --------=: HOST :=--------" << std::endl));
        #if defined(GET_TIME)
            Timer<HOST> TM(1, 22);//, StreamModifier::FG_CYAN
            TM.start();
        #endif

        int* Similarity_matrix = new int[Query.V * Target.V]();

        __ENABLE(PRINT_INFO, __PRINT("\t--------=: Similarity :=--------"
                                    "\t\tBFS Similarity Deep:\t" << BFS_SIMILARITY_DEEP << std::endl));

        similarity(Similarity_matrix);

        #if defined(GET_TIME)
            TM.getTime("Similarity");
            TM.start();
        #endif

        int* QuerySeeds = new int[NOF_SOLUTIONS];
        int* TargetSeeds = new int[NOF_SOLUTIONS];

        __ENABLE(PRINT_INFO, __PRINT("\t   --------=: Seed :=--------" << std::endl))
        seed(Similarity_matrix, QuerySeeds, TargetSeeds);

        delete[] Similarity_matrix;

        #if defined(GET_TIME)
            TM.getTime("Seed");
            TM.start();
        #endif

        __ENABLE(PRINT_INFO, __PRINT("\t  --------=: Extend :=--------" << std::endl));
        extend(QuerySeeds, TargetSeeds);

        #if defined(GET_TIME)
            TM.getTime("Extend");
        #endif

        delete[] QuerySeeds;
        delete[] TargetSeeds;
    }

    float* approx_subgraph_iso::LabelSimilarityMatrix = NULL;

    void approx_subgraph_iso::readSimilarityMatrix(const char* const File,
                                                   const int query_nodes,
                                                   const int target_nodes) {

        __ENABLE(PRINT_INFO, std::cout << "Similarity Matrix Reading..." << std::endl;)

        std::ifstream matrixFile(File);
        fileUtil::checkRegularFile(matrixFile);

        LabelSimilarityMatrix = new float[query_nodes * target_nodes];
        float* matrixLabelTMP = LabelSimilarityMatrix;
        for (int i = 0; i < query_nodes * target_nodes; i++)
            matrixFile >> *matrixLabelTMP++;

        matrixFile.close();
    }

    void approx_subgraph_iso::calculateCost() {
    	float Costs[NOF_SOLUTIONS];
    	const int TotalCost = Query.V + Query.V * Query.V;

    	float sum = 0, square = 0, maxValue = std::numeric_limits<float>::min(),
                                   minValue = std::numeric_limits<float>::max();
    	int nof_valid_solution = 0;
    	for (int K = 0; K < NOF_SOLUTIONS; K++) {
    		if (ValidSolutions[K]) {
    			int cost = 0;
    			for (int i = 0; i < Query.V; i++) {
    				if (Query.Labels[QueryMatch[K * Query.V + i]] != Target.Labels[TargetMatch[K * Query.V + i]])
    					cost++;
    			}
    			for (int i = 0; i < Query.V; i++) {
    				for (int j = 0; j < Query.V; j++) {
    					bool edgeOnQuery = isEdge<q_node_t, q_edge_t>
    						(Query.Nodes, Query.Edges, QueryMatch[K * Query.V + i], QueryMatch[K * Query.V + j]);
    					bool edgeOnTarget = isEdge<t_node_t, t_edge_t>
    						(Target.Nodes, Target.Edges, TargetMatch[K * Query.V + i], TargetMatch[K * Query.V + j]);
    					cost += edgeOnQuery ^ edgeOnTarget;
    				}
    			}
    			Costs[K] = (float) cost / TotalCost;
    			sum += Costs[K];
    			square += Costs[K] * Costs[K];
    			maxValue = std::max(maxValue, Costs[K]);
    			minValue = std::min(minValue, Costs[K]);
    			nof_valid_solution++;
    		}
    	}
    	const float devStd = std::sqrt(nof_valid_solution * square - sum * sum) / nof_valid_solution;
    	const float avg = sum / nof_valid_solution;

    	std::cout.precision(2);
    	std::cout	<< std::endl
                    << "  Requested solutions:   " << NOF_SOLUTIONS	     << std::endl
    				<< "    Founded solutions:   " << nof_valid_solution << std::endl << std::endl
    				<< "                  min:   " << minValue		     << std::endl
    				<< "                  max:   " << maxValue		     << std::endl
    				<< "                  avg:   " << avg			     << std::endl
    				<< "                  dev:   " << devStd		     << std::endl << std::endl;
    }


    void approx_subgraph_iso::uniqueSolutions() {
    	std::set< std::vector<q_node_t> > QuerySet;
    	for (int K = 0; K < NOF_SOLUTIONS; K++) {
    		if (ValidSolutions[K]) {
    			std::vector<q_node_t> QVector(&(QueryMatch[K * Query.V]), &(QueryMatch[K * Query.V]) + Query.V);
    			QuerySet.insert(QVector);
    		}
    	}

    	std::set< std::vector<t_node_t> > TargetSet;
    	for (int K = 0; K < NOF_SOLUTIONS; K++) {
    		if (ValidSolutions[K]) {
    			std::vector<t_node_t> TVector(&(TargetMatch[K * Query.V]), &(TargetMatch[K * Query.V]) + Query.V);
    			TargetSet.insert(TVector);
    		}
    	}
    	std::cout 	<< "  Query N. distinct solutions:\t" << QuerySet.size() << std::endl
    				<< " Target N. distinct solutions:\t" << TargetSet.size() << std::endl << std::endl;
    }

    void approx_subgraph_iso::printSolutions() {
        const int BUFFER_SIZE = MAX_SOLUTIONS * (2 + 11 * 256);
        // {}      : 2 * NOF_SOLUTIONS
        // ,()     : 3 * NOF_SOLUTIONS * |Q|
        // Q-match : 3 * NOF_SOLUTIONS * |Q|    // 256
        // T-match : 5 * NOF_SOLUTIONS * |Q|    // 65536
        static_assert((BUFFER_SIZE >> 20) > (8 >> 20),
        PRINT_MSG("Stack size not sufficient. Suggestion: $ ulimit -s unlimited"));
        char BufferS[BUFFER_SIZE];
        char* Buffer = BufferS;

        for (int K = 0; K < NOF_SOLUTIONS; K++) {
            if (ValidSolutions[K]) {
                q_node_t* QueryMatch_tmp = &QueryMatch[K * Query.V];
                t_node_t* TargetMatch_tmp = &TargetMatch[K * Query.V];

                *Buffer++ = '{';
                for (int i = 0; i < Query.V; i++) {
                    *Buffer++ = '(';
                    parsing::fastIntToString(*QueryMatch_tmp++, Buffer);
                    *Buffer++ = ',';
                    parsing::fastIntToString(*TargetMatch_tmp++, Buffer);
                    *Buffer++ = ')';
                }
                *Buffer++ = '}';
                *Buffer++ = '\n';
            }
        }
        *Buffer++ = '\n';
        *Buffer = '\0';
        assert(Buffer < BufferS + BUFFER_SIZE);
        //#if __linux__
        //fputs(BufferS, stdout);
        //fwrite(BufferS, 1, BufferS - Buffer, stdout);
        //#else
            std::ios_base::sync_with_stdio(false);
            std::cout << BufferS;
            //file.rdbuf()->pubsetbuf(Buffer, N);
        //#endif
    }
} //@host
} //@appagato
