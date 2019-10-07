# CUDABC - Artificial Bee Colony on GPU

Implementation of [Artificial Bee Colony Algorithm under General Purpose Graphics Processing Unit](http://www.sbpo2017.iltc.br/pdf/169439.pdf) architeture.

## Usage

Define the parameters in "parameters.py" and run the main file

## Notes about the implementation

    1) When this code was created (2017), instead of generating the random variables inside each CUDA kernel, it was faster to generate using NumPy library and send them to the GPU;

    2) At the same time, Numba CUDA kernels were not capable to use kernels inside classes, this is why each problem was defined in a file without a basic structure

    3) This is not the version used in the paper, the original version in CUDA C was lost 

    4) After running main.py, when the program stops, one folder will be created with the algorithm statistical information
