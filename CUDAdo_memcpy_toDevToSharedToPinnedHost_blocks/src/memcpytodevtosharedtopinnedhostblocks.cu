////////////////////////////////////////////////////////////////////////////
//
// Copyright 2017 Antonio Carrasco Valero.  All rights reserved.
//
////////////////////////////////////////////////////////////////////////////

/* memcpytodevtosharedtopinnedhostblocks.cu
 * 201709050310
 *
 * Exercise copying memory from host to device, copy into shared memory, and from shared memory to global, and then back to PINNED memory on the host, scheduling the kernel in a variable number of blocks.
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



__device__ void
memcpytodevtohostKernel_throughSharedMem_fullcollision(int *g_idata, int *g_odata, int theIntsToCopyPerThread, int theIntsToCopyPerShared)
{

	extern __shared__ int sh_Ints[];

	int* sh_th_Ints = 0;


    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    const unsigned int block = blockIdx.x;

    const unsigned int blockstart = block * num_threads * theIntsToCopyPerThread;

    const unsigned int shstart = theIntsToCopyPerShared * tid;

    int aNumSharedLoopsFromGlobal = 0;
    int aNumSharedLoopsToGlobal   = 0;

    int aNumIntsToCopyPending = theIntsToCopyPerThread;
    while( aNumIntsToCopyPending > 0) {

		int aNumIntsToCopy = aNumIntsToCopyPending;
		if( aNumIntsToCopy > theIntsToCopyPerShared) {
			aNumIntsToCopy = theIntsToCopyPerShared;
		}

		for (unsigned int anIntIdx = 0; anIntIdx < aNumIntsToCopy; ++anIntIdx) {
			unsigned int aGlobalDataIndex = blockstart + (( anIntIdx + ( aNumSharedLoopsFromGlobal * theIntsToCopyPerShared)) * num_threads) + tid;
			unsigned int aSharedDataIndex = shstart + anIntIdx;
			sh_Ints[ aSharedDataIndex] = g_idata[ aGlobalDataIndex];

		}
		aNumSharedLoopsFromGlobal++;

		for (unsigned int anIntIdx = 0; anIntIdx < aNumIntsToCopy; ++anIntIdx) {
			unsigned int aGlobalDataIndex = blockstart + (( anIntIdx + aNumSharedLoopsToGlobal * theIntsToCopyPerShared)  * num_threads) + tid;
			unsigned int aSharedDataIndex = shstart + anIntIdx;
			g_odata[ aGlobalDataIndex] = sh_Ints[ aSharedDataIndex];
		}

		aNumSharedLoopsToGlobal++;

		aNumIntsToCopyPending -= aNumIntsToCopy;
    }
}


__device__ void
memcpytodevtohostKernel_throughSharedMem(int *g_idata, int *g_odata, int theIntsToCopyPerThread, int theIntsToCopyPerShared)
{

	extern __shared__ int sh_Ints[];

	int* sh_th_Ints = 0;


    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    const unsigned int block = blockIdx.x;

    const unsigned int blockstart = block * num_threads * theIntsToCopyPerThread;

    const unsigned int shstart = theIntsToCopyPerShared * tid;

    int aNumSharedLoopsFromGlobal = 0;
    int aNumSharedLoopsToGlobal   = 0;

    int aNumIntsToCopyPending = theIntsToCopyPerThread;
    while( aNumIntsToCopyPending > 0) {

		int aNumIntsToCopy = aNumIntsToCopyPending;
		if( aNumIntsToCopy > theIntsToCopyPerShared) {
			aNumIntsToCopy = theIntsToCopyPerShared;
		}

		for (unsigned int anIntIdx = 0; anIntIdx < aNumIntsToCopy; ++anIntIdx) {
			unsigned int aGlobalDataIndex = blockstart + (( anIntIdx + ( aNumSharedLoopsFromGlobal * theIntsToCopyPerShared)) * num_threads) + tid;
			unsigned int aSharedDataIndex = ( num_threads * anIntIdx) + tid;
			sh_Ints[ aSharedDataIndex] = g_idata[ aGlobalDataIndex];

		}
		aNumSharedLoopsFromGlobal++;

		for (unsigned int anIntIdx = 0; anIntIdx < aNumIntsToCopy; ++anIntIdx) {
			unsigned int aGlobalDataIndex = blockstart + (( anIntIdx + aNumSharedLoopsToGlobal * theIntsToCopyPerShared)  * num_threads) + tid;
			unsigned int aSharedDataIndex = ( num_threads * anIntIdx) + tid;
			g_odata[ aGlobalDataIndex] = sh_Ints[ aSharedDataIndex];
		}

		aNumSharedLoopsToGlobal++;

		aNumIntsToCopyPending -= aNumIntsToCopy;
    }
}

__global__ void
memcpytodevtohostKernel(int *g_idata, int *g_odata, int theIntsToCopyPerThread, int theIntsToCopyPerShared)
{
	memcpytodevtohostKernel_throughSharedMem( g_idata, g_odata, theIntsToCopyPerThread, theIntsToCopyPerShared);

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
do_memcpytodevtosharedtopinnedhostblocks_general(int argc, char **argv,
		unsigned int theIntsToCopyPerThread,
		unsigned int theIntsToCopyPerShared,
		unsigned int theTotalShared,
		unsigned int theNumThreads,
		unsigned int theNumBlocks,
		char *theTitle,
		void (*theInitInputFunct)(int *, unsigned int, unsigned int),
		int (*theCheckOutputFunct)( int *, unsigned int, unsigned int))
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);



    unsigned int total_shared         = theTotalShared;
    unsigned int intstocopy_perthread = theIntsToCopyPerThread;
    unsigned int intstocopy_pershared = theIntsToCopyPerShared;
    unsigned int num_threads          = theNumThreads;
    unsigned int num_blocks           = theNumBlocks;
    unsigned int total_threads        = num_blocks * num_threads;
    unsigned int num_ints             = total_threads * intstocopy_perthread;
    unsigned int mem_size             = num_ints * sizeof(int);

    printf("%s\n", theTitle);
    printf("total_shared=%u; intstocopy_perthread=%u; intstocopy_pershared=%u, num_threads=%u; num_blocks=%u; total_threads=%u; num_ints=%u; mem_size=%u\n",
    		total_shared, intstocopy_perthread, intstocopy_pershared,
    		num_threads, num_blocks, total_threads, num_ints, mem_size);


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
    memcpytodevtohostKernel<<< grid, threads,total_shared>>>(d_idata, d_odata, intstocopy_perthread, intstocopy_pershared);







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

    printf("Copy From Device time: %f (ms)\n", sdkGetTimerValue(&timercopyfromdev));


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





/* From CUDAsample_deviceQuery
Device 0: "GeForce GTX 770M"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    3.0
  Total amount of global memory:                 3010 MBytes (3156148224 bytes)
  ( 5) Multiprocessors, (192) CUDA Cores/MP:     960 CUDA Cores
  GPU Max Clock rate:                            797 MHz (0.80 GHz)
  Memory Clock rate:                             2004 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 393216 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = GeForce GTX 770M
 */





/* as per CUDAsample_deviceQuery
 * ( 5) Multiprocessors, (192) CUDA Cores/MP:     960 CUDA Cores
 */
// #define NUM_CORES 960

/* NUM_THREADS is 32 equal to Warp size in the device to avoid shared memory bank conflicts:
 *     a shared memory bank conflict happens when more than one thread access different addresses in the same memory block
 * Each block shall use 32 threads which fit the warp size on the device
 */
#define NUM_THREADS 4 * 32 //  128 threads in bloc: 4 warps of 32 threads per block

/* NUM_BLOCKS  30  = NUM_CORES 960 / NUM_THREADS 32
 * Each block runs one 32 thread warp
 * All blocks can be run in parallel with all threads active as there are enough warps of 32 threads for them
 *
 * But because accesses to global memory take hundreds of processor cycles,
 * most processors remain idle waiting for global memory access
 * because there are not other/enough warps in a block which could be scheduled to process while accesses from a warp are being retrieved from global memory
 */
#define NUM_BLOCKS  300

// #define NUMALLTHREADS  960 // NUM_THREADS * NUM_BLOCKS


/* Integers to fill a gigabyte
 */
// #define NUMALLINTS 268435456 // 1024 * 1024 * 1024 / 4

/*  How many ints shall be managed by each thread
 * 268435456 / 960 = 279620.26666666666
 * Let's round, i.e. (this is kind of arbitrary) to the width of the bus: 192 bits / 32 bits per int: round to 6 ints
 *   279618
 * 279618 * 960 = 268433280 approximately equal but smaller than 268435456 (only 2176 short of the num of ints in a gigabyte)
*/
#define INTSTOCOPY_PERTHREAD 6990 // 69904 // 139809 // 55923 // 27961 // 279618


/*
 * Shared memory total 49152 / ( 32 * 4 = 128 threads in block ) = 384 bytes of shared memory per thread = 96 ints
 */
// #define INTSTOCOPY_PERSHARED 96
#define INTSTOCOPY_PERSHARED 14


/*
 * Total amount of shared memory per block:       49152 bytes
 *
 */
#define TOTALSHARED  ( INTSTOCOPY_PERSHARED * NUM_THREADS * sizeof( int))





/* ***************************************
 * Obtained 100% global mem read and write efficiency, and 75% occupancy with values
 *
 * #define NUM_THREADS 32 * 4
 * #define NUM_BLOCKS  300
 * #define INTSTOCOPY_PERTHREAD 6990
 */



void initInputFunct_contiguousInThread( int *theH_idata, unsigned int theIntsToCopyPerThread, unsigned int theNumTreads) {

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



int checkOutputFunct_contiguousInThread( int *theH_odata, unsigned int theIntsToCopyPerThread, unsigned int theNumTreads) {

	 for (unsigned int anIntIdx = 0; anIntIdx < theIntsToCopyPerThread; ++anIntIdx)
	       {
	       	for (unsigned int aThreadIdx = 0; aThreadIdx < theNumTreads; ++aThreadIdx)
	       	    {
	       			int aDataIndex = ( anIntIdx * theNumTreads) + aThreadIdx;
	       			int anExpected = ( aThreadIdx * theIntsToCopyPerThread) + anIntIdx;
	       			int anActual   = theH_odata[ aDataIndex];
					if( !( anActual == anExpected)) {
	       	            printf("intIdx=%d; thread=%d;  h_odata[ %d] = %d NOT THE EXPECTED %d\n", anIntIdx, aThreadIdx, aDataIndex, anActual, anExpected);
	       	            return 0;
	       	        }
	       	    }
	       }

	 return 1;
}



void
do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread(int argc, char **argv)
{
	do_memcpytodevtosharedtopinnedhostblocks_general(argc, argv,
			(unsigned int) INTSTOCOPY_PERTHREAD,
			(unsigned int) INTSTOCOPY_PERSHARED,
			(unsigned int) TOTALSHARED,
			(unsigned int) NUM_THREADS,
			(unsigned int) NUM_BLOCKS,
			"do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread",
			&initInputFunct_contiguousInThread, &checkOutputFunct_contiguousInThread);
}


void
do_memcpytodevtohost(int argc, char **argv) {

	// cudaDeviceSetCacheConfig( cudaFuncCachePreferShared);


	do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread( argc, argv);

	// do_memcpytodevtosharedtopinnedhostblocks_contiguousInMemory( argc, argv);

    exit(EXIT_SUCCESS);

}

/*
 * /home/acv/Works/MDD/CUDAwk/CUDAnsight_wkspcs/CUDAnsight_wkspc01/CUDAdo_memcpy_toDevToSharedToPinnedHost_blocks/Debug/CUDAdo_memcpy_toDevToSharedToPinnedHost_blocks Starting...


GPU Device 0: "GeForce GTX 770M" with compute capability 3.0



do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread
intstocopy_perthread=27961; num_threads=320; num_blocks=30; total_threads=9600; num_ints=268425600; mem_size=1073702400
Alloc host for output time: 0.009000 (ms)
CUDA Alloc host for output time: 246.917999 (ms)
Copy To Device time: 108.968002 (ms)
GPU Processing time: 0.030000 (ms)
Copy From Device time: 321.652008 (ms)
Memory and processing time: 432.950012 (ms)
CPU Processing time: 585.534973 (ms)
do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread All output OK




do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread
intstocopy_perthread=55923; num_threads=160; num_blocks=30; total_threads=4800; num_ints=268430400; mem_size=1073721600
Alloc host for output time: 0.012000 (ms)
CUDA Alloc host for output time: 242.134995 (ms)
Copy To Device time: 105.742996 (ms)
GPU Processing time: 0.037000 (ms)
Copy From Device time: 322.511993 (ms)
Memory and processing time: 430.563995 (ms)
CPU Processing time: 581.942017 (ms)
do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread All output OK





do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread
intstocopy_perthread=139809; num_threads=64; num_blocks=30; total_threads=1920; num_ints=268433280; mem_size=1073733120
Alloc host for output time: 0.009000 (ms)
CUDA Alloc host for output time: 258.083008 (ms)
Copy To Device time: 105.412003 (ms)
GPU Processing time: 0.035000 (ms)
Copy From Device time: 642.007019 (ms)
Memory and processing time: 750.289001 (ms)
CPU Processing time: 583.151001 (ms)
do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread All output OK





do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread
intstocopy_perthread=69904; num_threads=128; num_blocks=30; total_threads=3840; num_ints=268431360; mem_size=1073725440
Alloc host for output time: 0.009000 (ms)
CUDA Alloc host for output time: 251.809998 (ms)
Copy To Device time: 105.420998 (ms)
GPU Processing time: 0.028000 (ms)
Copy From Device time: 379.854004 (ms)
Memory and processing time: 487.601013 (ms)
CPU Processing time: 590.114014 (ms)
do_memcpytodevtosharedtopinnedhostblocks_contiguousInThread All output OK

 *
 */
