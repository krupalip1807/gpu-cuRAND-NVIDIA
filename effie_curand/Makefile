################################################################################
#
# Copyright (c) 2017, NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

CUDA_SDK_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda

TEST 	 = kernel_example
CXX      = $(CUDA_SDK_PATH)/bin/nvcc
EXE	 = o
CUDA_SDK_INCLUDE  = $(CUDA_SDK_PATH)/include

all: build

build: $(TEST).cu
	$(CXX) -L$(CUDA_SDK_PATH)/lib64 -L/usr/local/cuda/lib -I/usr/local/cuda/include -I$(CUDA_SDK_INCLUDE) -o $(TEST).$(EXE) $<

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
