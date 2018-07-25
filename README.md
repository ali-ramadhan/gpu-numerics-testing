# gpu-numerics-testing
Repo of prototype code for GPU and numerics testing for ocean GFD. This repo is for experiments that examine coding strategies and performance pros/cons for future MITgcm numerics engine.
Some experiments being that have been discussed include

- Simple adveciton test

- Simple linear shallow water test

- Simple variable space mixing test

These tests and geared toward trying out some new coding styles and approaches including portably targeting accelerators (particularly GPU), different levels of abstraction (CUDA/OpenCL, C/modern Fortran/C++, [MOOSE](http://mooseframework.org), [PyFR](http://www.pyfr.org), [Dedalus](http://dedalus-project.org), ...?) and different numerics (finite volume, discontinuous Galerkin, hybridized discontinuous Galerkin, ...?) and meshing ideas. Aside from code utility and practicality for science, relative performance and meaningful scaling are questions of interest.
