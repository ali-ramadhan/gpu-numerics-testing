# gpu-numerics-testing
Repo of prototype code for GPU and numerics testing for ocean GFD. This repo is for experiments that examine coding strategies and performance pros/cons for future MITgcm numerics engine.
Some experiments being that have been discussed include

- Simple adveciton test

- Simple linear shallow water test

- Simple variable space mixing test

These tests and geared toward trying out some new coding styles and approaches including targeting accelerators (particularly GPU), different levels of abstraction (CUDA/OpenCL, C/modern Fortran, PyFR/http://dedalus-project.org style etc...) and differnt numerics (FV, DG, HDG etc....)
