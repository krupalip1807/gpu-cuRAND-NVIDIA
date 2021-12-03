# Link for Google Colaboratory project
- [cuRAND](https://colab.research.google.com/drive/1Q5eCXbPydMPICMdU3TCu3lVZHvOaXP_4?usp=sharing)

# Setting up the project
Steps to setup the cuRAND project on Google colab -
## To Completely uninstall any previous CUDA versions.To refresh the Cloud Instance of CUDA.
```
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update
```

## To install CUDA Version 9
```
!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2
```

## To check CUDA installation
```
!nvcc --version
```

## Command to install a small extension to run nvcc from the Notebook cells
```
!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git
```

## To load the extension
```
%load_ext nvcc_plugin
```

## To Create/Update a file
```c
%%cuda --name kernel_example.cu
/*
 * Program code
 */
```

## To compile the kernel_example.cu
```
%%shell
nvcc /content/src/kernel_example.cu -lcurand -o /content/src/kernel_example 
```

## To run the kernel_example
```
%%shell
./src/kernel_example
```

## Command Line Parameters
```
./kernel_example [-m] [-p] [SampleSize]
- XORWOW is default generator.
- **-m:** To use MRG32k3a generator
- **-p:** To use Philox4_32_10_t generator
- SampleSize: To customize the size of sample, default is 10000.
```
