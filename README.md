# gpu-cuRAND-NVIDIA
This project has implemented the Random number generation using the cuRAND library. It is implemented on Google Colab and Effie. 

[[_TOC_]]
## Quick Start

### Compiling/Running this Code
```
git clone https://gitlab.indstate.edu/kparmar/cs67005-s2021-curand.git
cd cs67005-s2021-curand/effie_curand
make
./kernel_example.o
```

### Files to Look at
- `effie_curand/kernel_example.cu:` Generating Random numbers using different generators- XORWOW, MRG32k3a, Philox_4x32_10

### Documentation about cuRAND
- [GPU Gems 3 (Chapter 37)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)
- [NVIDIA cuRAND](https://developer.nvidia.com/curand)
- [Device API Overview](https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview)

## Getting Started - _kernel_example.cu_
```c
/*
 * This program uses the device CURAND API to calculate what
 * proportion of pseudo-random ints have low bit set.
 * It then generates uniform results to calculate how many
 * are greater than .5.
 * It then generates  normal results to calculate how many
 * are within one standard deviation of the mean.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void setup_kernel(curandStateMRG32k3a *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(0, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = curand(&localState);
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_kernel(curandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = curand(&localState);
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(curandState *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    float x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = curand_uniform(&localState);
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(curandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    float x;
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = curand_uniform(&localState);
        /* Check if > .5 */
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(curandState *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = curand_normal2(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
	if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(curandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = curand_normal2(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
	if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_kernel(curandStateMRG32k3a *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = curand(&localState);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(curandStateMRG32k3a *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    double x;
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = curand_uniform_double(&localState);
        /* Check if > .5 */
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(curandStateMRG32k3a *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    double2 x;
    /* Copy state to local memory for efficiency */
    curandStateMRG32k3a localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = curand_normal2_double(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
	if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

int main(int argc, char *argv[])
{
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 64;
    const unsigned int totalThreads = threadsPerBlock * blockCount;

    unsigned int i;
    unsigned int total;
    curandState *devStates;
    curandStateMRG32k3a *devMRGStates;
    curandStatePhilox4_32_10_t *devPHILOXStates;
    unsigned int *devResults, *hostResults;
    bool useMRG = 0;
    bool usePHILOX = 0;
    int sampleCount = 10000;
    bool doubleSupported = 0;
    int device;
    struct cudaDeviceProp properties;

    /* check for double precision support */
    CUDA_CALL(cudaGetDevice(&device));

    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    if ( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3) ) {
        doubleSupported = 1;
    }

    /* Check for MRG32k3a option (default is XORWOW) */
    if (argc >= 2)  {
        if (strcmp(argv[1],"-m") == 0) {
            useMRG = 1;
            if (!doubleSupported){
                printf("MRG32k3a requires double precision\n");
                printf("^^^^ test WAIVED due to lack of double precision\n");
                return EXIT_SUCCESS;
            }
	}else if (strcmp(argv[1],"-p") == 0) {
        usePHILOX = 1;
    }
     	/* Allow over-ride of sample count */
        sscanf(argv[argc-1],"%d",&sampleCount);
    }

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(totalThreads, sizeof(int));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, totalThreads *
              sizeof(unsigned int)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, totalThreads *
              sizeof(unsigned int)));

    /* Allocate space for prng states on device */
    if (useMRG) {
        CUDA_CALL(cudaMalloc((void **)&devMRGStates, totalThreads *
                  sizeof(curandStateMRG32k3a)));
    }else if(usePHILOX) {
        CUDA_CALL(cudaMalloc((void **)&devPHILOXStates, totalThreads *
                  sizeof(curandStatePhilox4_32_10_t)));
    }else {
	CUDA_CALL(cudaMalloc((void **)&devStates, totalThreads *
                  sizeof(curandState)));
    }

    /* Setup prng states */
    if (useMRG) {
        setup_kernel<<<64, 64>>>(devMRGStates);
    }else if(usePHILOX)
    {
     	setup_kernel<<<64, 64>>>(devPHILOXStates);
    }else {
	setup_kernel<<<64, 64>>>(devStates);
    }

    /* Generate and use pseudo-random  */
    for(i = 0; i < 10; i++) {
        if (useMRG) {
            generate_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults);
        }else if (usePHILOX){
            generate_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);
    }else {
            generate_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
        }
    }
   /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads *
        sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < totalThreads; i++) {
        total += hostResults[i];
    }
    printf("Fraction with low bit set was %10.13f\n",
        (float)total / (totalThreads * sampleCount * 10.0f));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, totalThreads *
              sizeof(unsigned int)));

    /* Generate and use uniform pseudo-random  */
    for(i = 0; i < 10; i++) {
        if (useMRG) {
            generate_uniform_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults);
        }else if(usePHILOX) {
            generate_uniform_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);
    }else {
            generate_uniform_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
        }
    }

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads *
        sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < totalThreads; i++) {
        total += hostResults[i];
    }
    printf("Fraction of uniforms > 0.5 was %10.13f\n",
        (float)total / (totalThreads * sampleCount * 10.0f));
    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, totalThreads *
              sizeof(unsigned int)));

    /* Generate and use normal pseudo-random  */
    for(i = 0; i < 10; i++) {
        if (useMRG) {
            generate_normal_kernel<<<64, 64>>>(devMRGStates, sampleCount, devResults);
        }else if(usePHILOX) {
            generate_normal_kernel<<<64, 64>>>(devPHILOXStates, sampleCount, devResults);
    }else {
            generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
        }
    }

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads *
        sizeof(unsigned int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < totalThreads; i++) {
        total += hostResults[i];
    }
    printf("Fraction of normals within 1 standard deviation was %10.13f\n",
        (float)total / (totalThreads * sampleCount * 10.0f));

    /* Cleanup */
    if (useMRG) {
        CUDA_CALL(cudaFree(devMRGStates));
    }else if(usePHILOX)
    {
     	CUDA_CALL(cudaFree(devPHILOXStates));
    }else {
	CUDA_CALL(cudaFree(devStates));
    }
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}
```

## Functions and Datatypes used
Documentation for the functions and datatypes used from Cuda and cuRAND -

### CUDA Functions
- `cudaGetDevice:` To get the device on which the device code is executed by the active host thread.
- `cudaGetDeviceProperties:` To get the properties of device.
- `cudaMalloc:` To allocate the linear memory on device
- `cudaMemset:` To fill the memory with the some constant value.
- `cudaMemcpy:` To copy bytes of memory from host/device to device/host.
- `cudaFree:` To free the allocated memory space.

### cuRAND Functions
- `curand_init:` Sets up an initial state. Allocate the given seed, sequence number, and offset within the sequence.
- `curand:` Returns a sequence of pseudorandom numbers.
- `curand_uniform:` Return quasirandom floats/doubles that ranges from 0.0 to 1.0. Here, 1.0 is included and 0.0 is excluded.
- `curand_normal2:` Return two normally distributed floats, that has mean 0.0f and standard deviation 1.0f from
the chosen generator in state.
- `curand_uniform_double:` Return a uniformly distributed double that lies between 0.0 and 1.0 from the chosen
generator in state,
- `curand_normal2_double:` Return two normally distributed doubles, that has mean 0.0 and standard deviation 1.0 from
the chosen generator in state.

### cuRAND Datatypes
- `curandState:` To implement the default XORWOW generator.
- `curandStateMRG32k3a:` To implement the MRG32k3a generator.
- `curandStatePhilox4_32_10_t:` To implement the Philox4_32_10_t generator.

## Link to Google Colaboratory
- [cuRAND on Colab](https://colab.research.google.com/drive/1Q5eCXbPydMPICMdU3TCu3lVZHvOaXP_4?usp=sharing)
