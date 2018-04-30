////////////////////////////////////////////////////////////////////////////
//
// Copyright 2017 Antonio Carrasco Valero.  All rights reserved.
//
////////////////////////////////////////////////////////////////////////////

/* memcpystreamsonehostthread.cu
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
#include <cuda_profiler_api.h>


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
			aDataIndex = blockstart + ( anIntIdx * num_threads) + tid;
			int aDataValue          = g_idata[ aDataIndex];
			g_odata[ aDataIndex] = aDataValue;
        }
}


__global__ void
memcpytodevtohostKernel(int *g_idata, int *g_odata, int theIntsToCopyPerThread)
{
	memcpytodevtohostKernel_coalescing( g_idata, g_odata, theIntsToCopyPerThread);
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
do_memcpystreamsonehostthread_general(int argc, char **argv,
		int theIntsToCopyPerThread, int theNumThreads,
		int theNumBatches, int theBlocksPerPatch,
		char *theTitle,
		void (*theInitInputFunct)(int *, int, int), int (*theCheckOutputFunct)( int *, int, int))
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);



    unsigned int intstocopy_perthread = theIntsToCopyPerThread;
    unsigned int num_threads          = theNumThreads;
    unsigned int num_batches          = theNumBatches;
    unsigned int blocks_per_batch     = theBlocksPerPatch;
    unsigned int num_ints             = num_batches * blocks_per_batch * num_threads * intstocopy_perthread;
    unsigned int mem_size             = num_ints * sizeof(int);

    printf("%s\n", theTitle);
    printf("intstocopy_perthread=%u; num_threads=%u; num_batches=%u; blocks_per_batch=%u; num_ints=%u; mem_size=%u\n",
    		intstocopy_perthread, num_threads, num_batches, blocks_per_batch, num_ints, mem_size);


    int *h_idata = 0;
    int *h_odata = 0;

    // allocate mem for the result on host side
    checkCudaErrors( cudaMallocHost( &h_idata, num_ints * sizeof( int)));



    (*theInitInputFunct)( h_idata, intstocopy_perthread, num_threads);




    // allocate mem for the result on host side
    checkCudaErrors( cudaMallocHost( &h_odata, num_ints * sizeof( int)));


    cleanOuput( h_odata, intstocopy_perthread, num_threads);






    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // allocate device memory
    int *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));




    // allocate device memory for result
    int *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));






    // cudaStream_t *someStreams = ( cudaStream_t *) malloc( num_batches * sizeof( cudaStream_t));
    cudaStream_t someStreams[ num_batches];


    // setup execution parameters
    dim3  grid( blocks_per_batch, 1, 1);
    dim3  threads( num_threads, 1, 1);




    unsigned int aNumIntsPerBatch      = blocks_per_batch * num_threads * intstocopy_perthread;
    unsigned int anIntsPerBatchMemsize = aNumIntsPerBatch * sizeof( int);

    for( int aBatchIdx=0; aBatchIdx < num_batches; aBatchIdx++) {
    	cudaStreamCreate( &someStreams[ aBatchIdx]);
    	cudaStream_t aStream = someStreams[ aBatchIdx];

    	unsigned int aBatch_idata_Idx = aBatchIdx * aNumIntsPerBatch;
    	int *h_idata_batch = &h_idata[ aBatch_idata_Idx];
    	int *d_idata_batch = &d_idata[ aBatch_idata_Idx];

    	// checkCudaErrors( cudaMemcpyAsync(d_idata_batch, h_idata_batch, anIntsPerBatchMemsize, cudaMemcpyHostToDevice, aStream));
    	cudaMemcpyAsync(d_idata_batch, h_idata_batch, anIntsPerBatchMemsize, cudaMemcpyHostToDevice, aStream);


		memcpytodevtohostKernel<<< grid, threads, 0, aStream>>>(d_idata, d_odata, intstocopy_perthread);

		// check if kernel execution generated and error
		// getLastCudaError("Kernel execution failed");

    	// int *h_odata_batch = &h_odata[ aBatch_idata_Idx];
		// int *d_odata_batch = &d_odata[ aBatch_idata_Idx];

	    // copy result from device to host
	    // checkCudaErrors( cudaMemcpyAsync(h_odata_batch, d_odata_batch, anIntsPerBatchMemsize, cudaMemcpyDeviceToHost, aStream));
		// cudaMemcpyAsync(h_odata_batch, d_odata_batch, anIntsPerBatchMemsize, cudaMemcpyDeviceToHost, aStream);
   }


    for( int aBatchIdx=0; aBatchIdx < num_batches; aBatchIdx++) {
    	cudaStream_t aStream = someStreams[ aBatchIdx];
		checkCudaErrors( cudaStreamSynchronize( aStream));
    }


    /*
    for( int aBatchIdx=0; aBatchIdx < num_batches; aBatchIdx++) {
    	cudaStream_t aStream = someStreams[ aBatchIdx];

    	unsigned int aBatch_idata_Idx = aBatchIdx * aNumIntsPerBatch;

    	int *h_odata_batch = &h_odata[ aBatch_idata_Idx];
    	int *d_odata_batch = &d_odata[ aBatch_idata_Idx];

	    // copy result from device to host
	    // checkCudaErrors( cudaMemcpyAsync(h_odata_batch, d_odata_batch, anIntsPerBatchMemsize, cudaMemcpyDeviceToHost, aStream));
	    cudaMemcpyAsync(h_odata_batch, d_odata_batch, anIntsPerBatchMemsize, cudaMemcpyDeviceToHost, aStream);
   }
   */


    for( int aBatchIdx=0; aBatchIdx < num_batches; aBatchIdx++) {
    	cudaStream_t aStream = someStreams[ aBatchIdx];
		checkCudaErrors( cudaStreamSynchronize( aStream));
		checkCudaErrors( cudaStreamDestroy( aStream));
    	// cudaStreamDestroy( aStream)
    }


    cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);


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

    checkCudaErrors( cudaFreeHost( h_idata));
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
// #define NUM_BATCHES  120
// #define BLOCKSPERBATCH  10
#define NUM_BATCHES  30
#define BLOCKSPERBATCH  10
// #define NUM_BLOCKS  ( NUM_BATCHES * BLOCKSPERBATCH)
#define NUM_THREADS 4 * 32
#define INTSTOCOPY_PERTHREAD 6976







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
do_memcpystreamsonehostthread(int argc, char **argv)
{
	do_memcpystreamsonehostthread_general(argc, argv, INTSTOCOPY_PERTHREAD, NUM_THREADS,
			NUM_BATCHES, BLOCKSPERBATCH,
			"do_memcpystreamsonehostthread",
			&initInputFunct_contiguousInThread, &checkOutputFunct_contiguousInThread);
}


void
do_memcpytodevtohost(int argc, char **argv) {

	do_memcpystreamsonehostthread( argc, argv);

	cudaProfilerStop();

    exit(EXIT_SUCCESS);

}

