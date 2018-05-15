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
using namespace timer;

#include "Host/GraphAPP.hpp"
#include "Host/approx_subgraph_iso.hpp"
#if defined(__DEVICE__)
    #include "Device/approx_subgraph_iso.cuh"
#endif

int main(int argc, char** argv) {
	if (argc != 4 && argc != 5)
		__ERROR("Invalid number of arguments: " << argv[0]
        << "  <TARGET> <QUERY> <N_OF_SOLUTIONS> [ Label_Similarity_Matrix ]")
	// ----------------- READ HEADER -------------------------------------------
    char *end;
	const int NOF_SOLUTIONS = std::strtol(argv[3], &end, 10);

    Timer<HOST> TMt(1, 22); //, StreamModifier::FG_CYAN
    TMt.start();

    #if defined(GET_TIME)
        Timer<HOST> TM(1, 22); //, StreamModifier::FG_CYAN
        TM.start();
    #endif

    int query_nodes, query_edges, target_nodes, target_edges;
    graph::readHeader(argv[1], target_nodes, target_edges);
    graph::readHeader(argv[2], query_nodes, query_edges);
    if (argc == 5)
        appagato::host::approx_subgraph_iso::readSimilarityMatrix(argv[4], query_nodes, target_nodes);

	// ----------------- CONTROLS ----------------------------------------------

	if (query_nodes > MAX_QUERY_SIZE)
		__ERROR("N. of Query Nodes > Max Query Nodes of " << MAX_QUERY_SIZE);
	if (query_edges > std::numeric_limits<q_edge_t>::max())
		__ERROR("N. of Query Edges (" << std::numeric_limits<q_edge_t>::max() << ") > Max value of " << fUtil::typeString<q_edge_t>());
	if (target_nodes > std::numeric_limits<t_node_t>::max())
		__ERROR("N. of Target Nodes (" << std::numeric_limits<t_node_t>::max() << ") > Max value of " << fUtil::typeString<t_node_t>());
	if (static_cast<unsigned>(target_edges) > std::numeric_limits<t_edge_t>::max())
		__ERROR("N. of Target Edges (" << std::numeric_limits<t_edge_t>::max() << ") > Max value of " << fUtil::typeString<t_edge_t>());
    if (NOF_SOLUTIONS > MAX_SOLUTIONS)
        __ERROR("N. of Solutions > MAX_SOLUTIONS (" << MAX_SOLUTIONS << ")")
    // ----------------- GRAPH  ------------------------------------------------

    graph::GraphAPP<q_node_t, q_edge_t> Query(query_nodes, query_edges);
    graph::GraphAPP<t_node_t, t_edge_t> Target(target_nodes, target_edges);

    // ----------------- READ GRAPH  -------------------------------------------

    graph::LabelMap_t LabelMap;
    Target.read<graph::G_TYPE::TARGET>(argv[1], &LabelMap);
    Query.read<graph::G_TYPE::QUERY>(argv[2], &LabelMap);

    #if defined(GET_TIME)
        TM.getTime("Read Time");
    #endif

    if (LabelMap.size() > MAX_LABELS)
        __ERROR("N. of labels (" << LabelMap.size() << ") > 256")
    __ENABLE(CHECK_ERROR, if (Query.isWeaklyConnected()) { __ERROR("Query graph is not weakly connected") })
    __ENABLE(PRINT_INFO, std::cout << "N. of Labels:\t" << LabelMap.size() << std::endl << std::endl;)
    __ENABLE(EXTRA_INFO, std::cout << "Query max frontier: " << std::endl; Query.maxFrontiers();)
    __ENABLE(EXTRA_INFO, std::cout << "Target max frontier: " << std::endl; Target.maxFrontiers();)

	// ------------------------ GRAPH ALGORITHM -----------------------------

    #if defined(__DEVICE__)
        appagato::device::approx_subgraph_iso app_problem( Query, Target, NOF_SOLUTIONS );
    #else
        appagato::host::approx_subgraph_iso app_problem( Query, Target, NOF_SOLUTIONS );
    #endif

	app_problem.resolve();

    __ENABLE(PRINT_INFO, app_problem.calculateCost();)
    __ENABLE(PRINT_INFO, app_problem.uniqueSolutions();)

    #if defined(GET_TIME)
        TM.start();
    #endif

    app_problem.printSolutions();

    #if defined(GET_TIME)
        TM.getTime("Write Time");
    #endif
    TMt.getTime("Total Time");
}
