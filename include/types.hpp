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

#include <limits>
#include <type_traits>
#include "../config.h"

const int  Q_NODE_SIZE  =  1;   //256
const int  Q_EDGE_SIZE  =  2;   //65536

const int  T_NODE_SIZE  =  2;   //65536
const int T_EDEGE_SIZE  =  4;   //2^32

const int   LABEL_SIZE  =  1;   //256

//------------------------------------------------------------------------------

template<int SIZE>
struct single_type {
    using type = typename std::conditional<SIZE == 1,
                    unsigned char,
                    typename std::conditional<SIZE == 2, unsigned short, unsigned int>::type
                >::type;
};
using q_node_t = typename single_type<Q_NODE_SIZE>::type;

using q_edge_t = typename single_type<Q_EDGE_SIZE>::type;

using t_node_t = typename single_type<T_NODE_SIZE>::type;

using t_edge_t = typename single_type<T_EDEGE_SIZE>::type;

using label_t  = typename single_type<LABEL_SIZE>::type;


const int    MAX_QUERY_SIZE  =  std::numeric_limits<q_node_t>::max() - 2;
const int        MAX_LABELS  =  std::numeric_limits<label_t>::max();
const q_node_t    INF_QNODE  =  std::numeric_limits<q_node_t>::max() - 1;
const q_node_t     IN_QUEUE  =  std::numeric_limits<q_node_t>::max() - 2;

#if defined(__DEVICE__)
    #include <vector_types.h>

    using mark_t = typename std::conditional<CAS, unsigned int, q_node_t>::type;

    template<int SIZE>
    struct double_type {
        using type = typename std::conditional<T_NODE_SIZE == 1,
                        uchar2,
                        typename std::conditional<T_NODE_SIZE == 2, ushort2, uint2>::type
                    >::type;
    };

    using q_node_t2 = typename double_type<Q_NODE_SIZE>::type;

    using q_edge_t2 = typename double_type<Q_EDGE_SIZE>::type;

    using t_node_t2 = typename double_type<T_NODE_SIZE>::type;
#else
    using mark_t = q_node_t;
#endif
