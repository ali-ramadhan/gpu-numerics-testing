# GPU test problem #1: One-dimensional wave equation

Code for stepping forward a one-dimensional wave equation on a domain with N points and with M instances. The wave equation is discretized using finite differences. Each instance is initialized with a Gaussian distribution with random mean and standard deviation. Experiments with single CPU, multicore CPU, CUDA, and openACC. Maybe PyFR.

## How to install software and libraries needed on the Engaging Cluster

I run with the latest version of the CUDA toolkit, gcc, and Python3 (to animate PDE output over X11). So you'll want to load the following modules
<pre>  1) cuda75/toolkit/7.5.18   2) engaging/python/3.6.0   3) gcc/4.8.4</pre>

### Eigen3

The CPU C++ tests rely on the [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) template library for linear algebra functions. No need to install it, just download the source code and include it when building:
<pre>g++ -I /path/to/eigen/ my_program.cpp -o my_program</pre>

### CUDA

Some of the CUDA C++ code relies on the sample code provided by the toolkit. To obtain the samples, run
<pre>cuda-install-samples-7.5.sh ~/cuda-samples/</pre>
which can then be included when compiling.

### OpenACC

Quoting the [Compute Canada OpenACC tutorial](https://docs.computecanada.ca/wiki/OpenACC_Tutorial_-_Adding_directives)
> As of May 2016, compiler support for OpenACC is still relatively scarce. Being pushed by NVidia, through its Portland Group division, as well as by Cray, these two lines of compilers offer the most advanced OpenACC support. GNU Compiler support for OpenACC exists, but is considered experimental in version 5. It is expected to be officially supported in version 6 of the compiler.
>
> For the purpose of this tutorial, we use version 16.3 of the Portland Group compilers. We note that Portland Group compilers are free for academic usage.

So I use PGI's <tt>pgc++</tt> to compile OpenACC code. <b>BUT</b>, the latest version offered on Engaging is <tt>engaging/pgi/15.7</tt> which is almost 3 years old, does not support C++11, and gave me issues on the reserved GPU node (but not on regular nodes). So I created a local installation of version 18.4 in <tt>~/.pgi/</tt> which works.

You'll want to download the [PGI Community Edition](https://www.pgroup.com/products/community.htm) and follow the [installation steps for Linux](https://www.pgroup.com/resources/docs/18.4/x86/pgi-install-guide/index.htm#install-linux-steps). It's a very simple install although it takes up 6.5GB so keep that in mind.

## How to compile tests

To build the single CPU test:
<pre>g++ --std=c++11 -Wall -I ~/eigen3/ cpu_wave_equation_1D.cpp -o cpu_wave_equation_1D</pre>

To build the CUDA test (to run on NVIDIA Tesla P100):
<pre>nvcc --std=c++11 -g -G -lcusolver -lcublas -lcusparse -I ~/NVIDIA_CUDA-7.5_Samples/common/inc/ cuda_wave_equation_1D.cu -o cuda_wave_equation_1D</pre>

To build the CPU multicore OpenACC test:
<pre>pgc++ -v -std=c++11 -fast -Minfo -ta=host -I ~/eigen3/ openacc_multicore_wave_equation_1d.cpp -o openacc_multicore_wave_equation_1d</pre>
