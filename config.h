#pragma once

const int MAX_SOLUTIONS = 1000;

// ===================  BFS SIMILARITY =========================================

const bool LOW_MEMORY = true;
	     // semi-deterministic   //CAS = 1 check correcteness with host

const int BFS_SIMILARITY_DEEP = 2;

// KERNEL CONFIG

const int BLOCKDIM_SIM_QUERY = 128;
const int MIN_VW_SIM_QUERY  =  32;
const int MAX_VW_SIM_QUERY  =  32;

const int BLOCKDIM_SIM_TARGET  =  512;
const int   MIN_VW_SIM_TARGET  =  32;
const int   MAX_VW_SIM_TARGET  =  32;

// ================ ALL KERNELS ================================================

const int MAX_REG_ARRAY_SIZE = 	50;


// EXTEND ----------------------------------------------------------------------

const int BLOCKDIM_EXT = 512;
const int   MIN_VW_EXT = 32;
const int   MAX_VW_EXT = 32;

const int NOF_PROBS_REGISTER  = 20;

//------------------------------------------------------------------------------

#if !defined(CHECK_ERROR)
    #define CHECK_ERROR 0
#endif
#if !defined(PRINT_INFO)
    #define PRINT_INFO 0
#endif
#if !defined(EXTRA_INFO)
    #define EXTRA_INFO 0
#endif
#if !defined(DEBUG_READ)
    #define DEBUG_READ 0
#endif
#if defined(__DEVICE__)
    #if !defined(CHECK_RESULT)
        #define CHECK_RESULT 0
        #define          CAS 1
    #else
        #define          CAS 0
    #endif
#endif
