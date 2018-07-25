# GPU test problem #1: One-dimensional wave equation

Code for stepping forward a one-dimensional wave equation on a domain with N points and with M instances. The wave equation is discretized using finite differences. Each instance is initialized with a Gaussian distribution with random mean and standard deviation. Experiments with single CPU, multicore CPU, CUDA, and openACC. Maybe PyFR.

Some of the C++ tests rely on the [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) template library for linear algebra. No need to install it, just download the source code and include it when building: <pre>g++ -I /path/to/eigen/ my_program.cpp -o my_program</pre>

Some of the CUDA C++ code relies on the sample code provided by the toolkit. To obtain the samples, run:
<pre>cuda-install-samples-7.5.sh ~/cuda-samples/</pre>

To build the single CPU code:
<pre>g++ --std=c++11 -Wall -I ~/eigen3/ cpu_wave_equation_1D.cpp -o cpu_wave_equation_1D</pre>

To build the CUDA code (to run on NVIDIA Tesla P100):
<pre>nvcc --std=c++11 -g -G -lcusolver -lcublas -lcusparse -I ~/NVIDIA_CUDA-7.5_Samples/common/inc/ cuda_wave_equation_1D.cu -o cuda_wave_equation_1D</pre>
