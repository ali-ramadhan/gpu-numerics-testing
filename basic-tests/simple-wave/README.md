Code for stepping forward simple wave equation on a domain with N points and with M instances. Experiments with CUDA, PyFR, C, Fortran, OpenACC.

Some of the C++ tests rely on the [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) template library for linear algebra. No need to install it, just download the source code and include it when building: <pre>g++ -I /path/to/eigen/ my_program.cpp -o my_program</pre>.
