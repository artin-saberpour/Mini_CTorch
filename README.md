# Mini_Ctorch

**A neural-network implementation in C with custom CUDA kernels for GPU-accelerated computation and model training.**

Mini_Ctorch is a from-scratch deep-learning implementation inspired by PyTorch. I built it to work directly with the computational machinery behind neural networks, from low-level numerical operations and CUDA kernel programming to constructing network architectures and running training experiments.

Rather than only using a high-level machine-learning framework, this project involves implementing the computation that makes model execution and training possible.

## What I implemented

- **CUDA GPU programming:** Writing custom CUDA kernels for neural-network computations and working directly with GPU execution.
- **Matrix multiplication:** Implementing matrix multiplication as a core numerical building block for neural-network layers.
- **Neural-network construction:** Building network structures using the project's own implementation, rather than defining the entire model through an existing deep-learning framework.
- **Model training:** Running training experiments with the implemented network and computation pipeline.
- **C-level implementation:** Developing the underlying neural-network functionality in C, with CUDA for GPU-side computation.

Together, these components cover more than a training visualization. The project connects low-level GPU computation to network-level execution and learning.

## Training demonstration

The animation below visualizes the progress of a network trained using Mini_Ctorch.

![Neural-network training progress](py/training_progress.gif)

*Training progress is the visible result of the implementation; the main engineering work is in the CUDA kernels, matrix operations, network construction, and training code.*

## Engineering focus

The project provides hands-on experience across two levels of machine-learning engineering.

**GPU computation:** Expressing numerical operations as CUDA kernels, including matrix multiplication, and connecting those operations to neural-network workloads.

**Model implementation:** Constructing neural-network structures and executing training using the project's own lower-level components.

The emphasis is on understanding and implementing the stack beneath a framework-level API, not on reproducing every feature of PyTorch.

## Repository and reproducibility

The training animation is available at [`py/training_progress.gif`](py/training_progress.gif).

For implementation details, refer to the C and CUDA source files in the repository. This README does not yet specify build commands, API coverage, benchmark results, or hardware requirements because those details have not been verified from the source tree.

## Project scope

Mini_Ctorch is an independent implementation inspired by PyTorch, not a full PyTorch replacement. Its focus is hands-on GPU kernel programming, neural-network computation, model construction, and training.
