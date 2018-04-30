////////////////////////////////////////////////////////////////////////////
//
// Copyright 2017 Antonio Carrasco Valero.  All rights reserved.
//
////////////////////////////////////////////////////////////////////////////

/* memcpytodevtohost.cu
 * 201709042335
 *
 * Exercise copying memory from host to device and back, scheduling the kernel in just one block.
 *
 */

/* Started from Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include "helper_cuda.h"
#include "helper_functions.h" // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void do_memcpytodevtohost(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__device__ void
memcpytodevtohostKernel_coalescing(int *g_idata, int *g_odata, int theIntsToCopyPerThread)
{

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
        {
			unsigned int aDataIndex = 0;
			if( tid % 2) {
				aDataIndex = ( anIntIdx * num_threads) + tid;
			}
			else {
				aDataIndex = ( anIntIdx * num_threads) + tid;
			}
			int aDataValue          = g_idata[ aDataIndex];
			g_odata[ aDataIndex] = aDataValue;
        }
}

__device__ void
memcpytodevtohostKernel_notcoalescing(int *g_idata, int *g_odata, int theIntsToCopyPerThread)
{

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
        {
			unsigned int aDataIndex = 0;
			if( tid % 2) {
				aDataIndex = ( anIntIdx * num_threads) + tid;
			}
			else {
				aDataIndex = ( ( theIntsToCopyPerThread - anIntIdx - 1) * num_threads) + tid;
			}
			int aDataValue          = g_idata[ aDataIndex];
			g_odata[ aDataIndex] = aDataValue;
        }
}

__global__ void
memcpytodevtohostKernel(int *g_idata, int *g_odata, int theIntsToCopyPerThread)
{
	memcpytodevtohostKernel_coalescing( g_idata, g_odata, theIntsToCopyPerThread);

	// memcpytodevtohostKernel_notcoalescing( g_idata, g_odata, theIntsToCopyPerThread);

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	do_memcpytodevtohost(argc, argv);
}





void cleanOuput( int *theH_odata, int theIntsToCopyPerThread, int theNumTreads) {

    // initalize the memory
    for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
    {
    	for (unsigned int aThreadIdx = 0; aThreadIdx < theNumTreads; ++aThreadIdx)
    	    {
    			int aDataIndex = ( anIntIdx * theNumTreads) + aThreadIdx;
    			theH_odata[ aDataIndex] = 0;
    	    }
    }
}




void
do_memcpytodevtohost_general(int argc, char **argv,
		int theIntsToCopyPerThread, int theNumThreads,
		char *theTitle,
		void (*theInitInputFunct)(int *, int, int), int (*theCheckOutputFunct)( int *, int, int))
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);


    unsigned int intstocopy_perthread = theIntsToCopyPerThread;
    unsigned int num_threads          = theNumThreads;
    unsigned int mem_size             = theNumThreads * theIntsToCopyPerThread * sizeof(int);

    printf("%s\n", theTitle);
    printf("intstocopy_perthread=%d; num_threads=%d; mem_size=%d\n", intstocopy_perthread, num_threads, mem_size);


    // allocate host memory
    int *h_idata = (int *) malloc( mem_size);

    (*theInitInputFunct)( h_idata, intstocopy_perthread, num_threads);


    // allocate mem for the result on host side
    int *h_odata = (int *) malloc(mem_size);

    cleanOuput( h_odata, intstocopy_perthread, num_threads);


    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // allocate device memory
    int *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
                               cudaMemcpyHostToDevice));

    // allocate device memory for result
    int *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(num_threads, 1, 1);

    // execute the kernel
    memcpytodevtohostKernel<<< grid, threads>>>(d_idata, d_odata, intstocopy_perthread);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");


    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size,
                               cudaMemcpyDeviceToHost));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    int aCheckedOk = (*theCheckOutputFunct)( h_odata, intstocopy_perthread, num_threads);

    // cleanup memory
    free(h_idata);
    free(h_odata);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    if( aCheckedOk) {
		printf("%s All output OK\n", theTitle);
	}
	else {
		printf("%s Error. Exiting.\n", theTitle);
		exit( EXIT_FAILURE);
	}
}








#define NUM_THREADS 1024
#define INTSTOCOPY_PERTHREAD 65536 * 4




void initInputFunct_contiguousInMemory( int *theH_idata, int theIntsToCopyPerThread, int theNumTreads) {

    // initalize the memory
    for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
    {
    	for (unsigned int aThreadIdx = 0; aThreadIdx < theNumTreads; ++aThreadIdx)
    	    {
    			int aDataIndex = ( anIntIdx * theNumTreads) + aThreadIdx;
    			int aDataValue = ( aThreadIdx * theIntsToCopyPerThread) + anIntIdx;
    			theH_idata[ aDataIndex] = aDataValue;
    	    }
    }
}



int checkOutputFunct_contiguousInMemory( int *theH_odata, int theIntsToCopyPerThread, int theNumTreads) {

	 for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
	       {
	       	for (unsigned int aThreadIdx = 0; aThreadIdx < theNumTreads; ++aThreadIdx)
	       	    {
	       			int aDataIndex = ( anIntIdx * theNumTreads) + aThreadIdx;
	       			int anExpected = ( aThreadIdx * theIntsToCopyPerThread) + anIntIdx;
	       			int anActual   = theH_odata[ aDataIndex];
					if( !( anActual == anExpected)) {
	       	            printf("h_odata[ %d] = %d NOT THE EXPECTED %d\n", aDataIndex, anActual, anExpected);
	       	            return 0;
	       	        }
	       	    }
	       }

	 return 1;
}



void
do_memcpytodevtohost_contiguousInMemory(int argc, char **argv) {

	do_memcpytodevtohost_general(argc, argv, INTSTOCOPY_PERTHREAD, NUM_THREADS, "do_memcpytodevtohost_contiguousInMemory", &initInputFunct_contiguousInMemory, &checkOutputFunct_contiguousInMemory);
}







void initInputFunct_contiguousInThread( int *theH_idata, int theIntsToCopyPerThread, int theNumTreads) {

    // initalize the memory
	for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
		{
			for (unsigned int aThreadIdx = 0; aThreadIdx < theNumTreads; ++aThreadIdx)
				{
					int aDataIndex = ( anIntIdx * theNumTreads) + aThreadIdx;
					int aDataValue = ( aThreadIdx * theIntsToCopyPerThread) + anIntIdx;
					theH_idata[ aDataIndex] = aDataValue;
				}
		}
}



int checkOutputFunct_contiguousInThread( int *theH_odata, int theIntsToCopyPerThread, int theNumTreads) {

	 for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
	       {
	       	for (unsigned int aThreadIdx = 0; aThreadIdx < theNumTreads; ++aThreadIdx)
	       	    {
	       			int aDataIndex = ( anIntIdx * theNumTreads) + aThreadIdx;
	       			int anExpected = ( aThreadIdx * theIntsToCopyPerThread) + anIntIdx;
	       			int anActual   = theH_odata[ aDataIndex];
					if( !( anActual == anExpected)) {
	       	            printf("h_odata[ %d] = %d NOT THE EXPECTED %d\n", aDataIndex, anActual, anExpected);
	       	            return 0;
	       	        }
	       	    }
	       }

	 return 1;
}



void
do_memcpytodevtohost_contiguousInThread(int argc, char **argv)
{
	do_memcpytodevtohost_general(argc, argv, INTSTOCOPY_PERTHREAD, NUM_THREADS, "do_memcpytodevtohost_contiguousInThread",
			&initInputFunct_contiguousInThread, &checkOutputFunct_contiguousInThread);
}


void
do_memcpytodevtohost(int argc, char **argv) {
	do_memcpytodevtohost_contiguousInMemory( argc, argv);

	do_memcpytodevtohost_contiguousInThread( argc, argv);


	do_memcpytodevtohost_contiguousInMemory( argc, argv);

	do_memcpytodevtohost_contiguousInThread( argc, argv);


	// do_memcpytodevtohost_contiguousInMemory( argc, argv);

	// do_memcpytodevtohost_contiguousInThread( argc, argv);

    exit(EXIT_SUCCESS);

}

