////////////////////////////////////////////////////////////////////////////
//
// Copyright 2017 Antonio Carrasco Valero.  All rights reserved.
//
////////////////////////////////////////////////////////////////////////////

/* memcpytodevtopinnedhostblocks.cu
 * 201709050310
 *
 * Exercise copying memory from host to device and back to PINNED memory on the host, scheduling the kernel in a variable number of blocks.
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

    const unsigned int block = blockIdx.x;

    const unsigned int blockstart = block * num_threads * theIntsToCopyPerThread;


    for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
        {
			unsigned int aDataIndex = 0;
			if( tid % 2) {
				aDataIndex = blockstart + ( anIntIdx * num_threads) + tid;
			}
			else {
				aDataIndex = blockstart + ( anIntIdx * num_threads) + tid;
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

    const unsigned int block = blockIdx.x;

    const unsigned int blockstart = block * num_threads * theIntsToCopyPerThread;


    for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
        {
			unsigned int aDataIndex = 0;
			if( tid % 2) {
				aDataIndex = blockstart + ( anIntIdx * num_threads) + tid;
			}
			else {
				aDataIndex = blockstart + ( ( theIntsToCopyPerThread - anIntIdx - 1) * num_threads) + tid;
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
do_memcpytodevtopinnedhostblocks_general(int argc, char **argv,
		int theIntsToCopyPerThread, int theNumThreads, int theNumBlocks,
		char *theTitle,
		void (*theInitInputFunct)(int *, int, int), int (*theCheckOutputFunct)( int *, int, int))
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);


    unsigned int intstocopy_perthread = theIntsToCopyPerThread;
    unsigned int num_threads          = theNumThreads;
    unsigned int num_blocks           = theNumBlocks;
    unsigned int num_ints             = num_blocks * num_threads * intstocopy_perthread;
    unsigned int mem_size             = num_ints * sizeof(int);

    printf("%s\n", theTitle);
    printf("intstocopy_perthread=%d; num_threads=%d; num_blocks=%d; num_ints=%d; mem_size=%d\n", intstocopy_perthread, num_threads, num_blocks, num_ints, mem_size);


    // allocate host memory
    int *h_idata = (int *) malloc( mem_size);

    (*theInitInputFunct)( h_idata, intstocopy_perthread, num_threads);

    StopWatchInterface *timerallochost = 0;
    sdkCreateTimer(&timerallochost);
    sdkStartTimer(&timerallochost);

    // allocate mem for the result on host side
    int *h_odata = (int *) malloc(mem_size);


    sdkStopTimer(&timerallochost);

    printf("Alloc host for output time: %f (ms)\n", sdkGetTimerValue(&timerallochost));

    free(h_odata);


    sdkStartTimer(&timerallochost);

    // allocate mem for the result on host side
    checkCudaErrors( cudaMallocHost( &h_odata, num_ints * sizeof( int)));

    sdkStopTimer(&timerallochost);

    printf("CUDA Alloc host for output time: %f (ms)\n", sdkGetTimerValue(&timerallochost));



    cleanOuput( h_odata, intstocopy_perthread, num_threads);


    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // allocate device memory
    int *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));


    StopWatchInterface *timercopytodev = 0;
    sdkCreateTimer(&timercopytodev);
    sdkStartTimer(&timercopytodev);

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
                               cudaMemcpyHostToDevice));

    sdkStopTimer(&timercopytodev);

    printf("Copy To Device time: %f (ms)\n", sdkGetTimerValue(&timercopytodev));


    // allocate device memory for result
    int *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid(num_blocks, 1, 1);
    dim3  threads(num_threads, 1, 1);


    StopWatchInterface *timergpu = 0;
    sdkCreateTimer(&timergpu);
    sdkStartTimer(&timergpu);

    // execute the kernel
    memcpytodevtohostKernel<<< grid, threads>>>(d_idata, d_odata, intstocopy_perthread);

    sdkStopTimer(&timergpu);

    printf("GPU Processing time: %f (ms)\n", sdkGetTimerValue(&timergpu));


    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");



    StopWatchInterface *timercopyfromdev = 0;
    sdkCreateTimer(&timercopyfromdev);
    sdkStartTimer(&timercopyfromdev);


    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size,
                               cudaMemcpyDeviceToHost));


    sdkStopTimer(&timercopyfromdev);

    printf("Copy %u From Device time: %f (ms)\n", mem_size, sdkGetTimerValue(&timercopyfromdev));


    sdkStopTimer(&timer);
    printf("Memory and processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    int aCheckedOk = (*theCheckOutputFunct)( h_odata, intstocopy_perthread, num_threads);


    // cleanup device memory
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));


    StopWatchInterface *timercpu = 0;
    sdkCreateTimer(&timercpu);
    sdkStartTimer(&timercpu);

    /*
    for( int aHostIdx=0; aHostIdx < num_ints; aHostIdx++) {
    	h_odata[  aHostIdx] = h_idata[ aHostIdx];
    }
    */


    int *aHost_iptr = h_idata;
    int *aHost_optr = h_odata;


    for( int aHostIdx=0; aHostIdx < num_ints; aHostIdx++) {
		*aHost_optr = *aHost_iptr;
		aHost_optr++;
		aHost_iptr++;
	}

    sdkStopTimer(&timercpu);

    printf("CPU Processing time: %f (ms)\n", sdkGetTimerValue(&timercpu));


    // cleanup host memory
    free(h_idata);
    // free(h_odata);
    checkCudaErrors( cudaFreeHost( h_odata));

    if( aCheckedOk) {
		printf("%s All output OK\n", theTitle);
	}
	else {
		printf("%s Error. Exiting.\n", theTitle);
		exit( EXIT_FAILURE);
	}

}






// FASTER transfer from device to host
/* */
#define NUM_BLOCKS  42
#define NUM_THREADS 192
#define INTSTOCOPY_PERTHREAD 32768
/* */

// SLOWER transfer from device to host
/*
#define NUM_BLOCKS  30
#define NUM_THREADS 32
#define INTSTOCOPY_PERTHREAD 279618
*/




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
do_memcpytodevtopinnedhostblocks_contiguousInMemory(int argc, char **argv) {

	do_memcpytodevtopinnedhostblocks_general(argc, argv, INTSTOCOPY_PERTHREAD, NUM_THREADS, NUM_BLOCKS,
			"do_memcpytodevtopinnedhostblocks_contiguousInMemory",
			&initInputFunct_contiguousInMemory, &checkOutputFunct_contiguousInMemory);
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
do_memcpytodevtopinnedhostblocks_contiguousInThread(int argc, char **argv)
{
	do_memcpytodevtopinnedhostblocks_general(argc, argv, INTSTOCOPY_PERTHREAD, NUM_THREADS, NUM_BLOCKS,
			"do_memcpytodevtopinnedhostblocks_contiguousInThread",
			&initInputFunct_contiguousInThread, &checkOutputFunct_contiguousInThread);
}


void
do_memcpytodevtohost(int argc, char **argv) {

	do_memcpytodevtopinnedhostblocks_contiguousInThread( argc, argv);

	// do_memcpytodevtopinnedhostblocks_contiguousInMemory( argc, argv);

    exit(EXIT_SUCCESS);

}

