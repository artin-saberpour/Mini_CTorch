# Mini_CTorch

**A neural-network and automatic-differentiation implementation in C and CUDA.**

Mini_CTorch implements tensor operations, custom CUDA kernels, gradient propagation, fully connected layers, and parameter optimization. The example below trains a GPU-resident network to approximate a sinusoidal target, without using PyTorch or another neural-network framework for the computation.

## Training demonstration

<p align="center">
  <img src="py/training_progress.gif" alt="Training progress of Mini_CTorch's sine-function approximation" width="740">
</p>

The animation visualizes predicted values against the target during optimization. The example is defined in [`main_network_sin.c`](c/code/examples/main_network_sin.c), with its configuration in [`params.txt`](c/params.txt).

### Network architecture

The current example uses **three fully connected layers**, two `tanh` activations, and a scalar output.

```text
  Input          Hidden 1           Hidden 2          Output
    x       ──►  Linear(1, 8) ──► tanh ──► Linear(8, 8) ──► tanh ──► Linear(8, 1) ──► ŷ
                  W₁, b₁                      W₂, b₂                      W₃, b₃
```

Mathematically,

$$
\begin{aligned}
h_1 &= \tanh(xW_1+b_1),\\
h_2 &= \tanh(h_1W_2+b_2),\\
\hat y &= h_2W_3+b_3.
\end{aligned}
$$

The number of layers and hidden width are configurable in `c/params.txt`; the diagram reflects the **checked-in configuration**, not a fixed library limitation.

### Ground truth and training objective

The example creates **50 training samples**, indexed by \(i=0,\ldots,49\):

$$
x_i=\frac{2i}{50}-1,\qquad
y_i^{\mathrm{GT}}=\sin\left(\frac{2\pi i}{50}\right).
$$

Equivalently, at these sampled inputs, the underlying target is

$$
\boxed{f_{\mathrm{GT}}(x)=\sin\bigl(\pi(x+1)\bigr).}
$$

The loss implemented in [`loss.c`](c/code/src/loss.c) is mean squared error:

$$
\boxed{\mathcal L(\theta)=\frac{1}{50}\sum_{i=0}^{49}
\left(f_\theta(x_i)-y_i^{\mathrm{GT}}\right)^2.}
$$

Training uses gradient descent with gradients propagated through the tensor computation graph:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal L(\theta_t).
$$

| Demonstration setting | Value |
|---|---|
| Network | `1 → 8 → 8 → 1` |
| Hidden activation | `tanh` |
| Training samples | `50` |
| Optimizer | SGD |
| Learning rate | `0.1` |
| Epochs | `2001` |
| Compute device | CUDA |

The training loop writes the loss to `loss_data.csv` and prediction snapshots every 10 epochs for visualization.

## CUDA implementation

The project implements GPU-side numerical primitives used in **both the forward and backward passes**, rather than merely calling GPU-enabled framework layers.

### Matrix multiplication

For \(A\in\mathbb R^{m\times k}\) and \(B\in\mathbb R^{k\times n}\),

$$
C_{ij}=\sum_{p=0}^{k-1}A_{ip}B_{pj}.
$$

The CUDA implementation maps output elements to GPU threads in a **16 × 16 thread-block layout**. Its backward kernels compute

$$
\frac{\partial\mathcal L}{\partial A}
=\frac{\partial\mathcal L}{\partial C}B^\top,
\qquad
\frac{\partial\mathcal L}{\partial B}
=A^\top\frac{\partial\mathcal L}{\partial C}.
$$

A representative excerpt from [`ops_matmul_cuda.cu`](c/code/src/ops_matmul_cuda.cu):

```cuda
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int k, int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int p = 0; p < k; ++p)
        sum += A[row * k + p] * B[p * n + col];

    C[row * n + col] = sum;
}
```

The repository additionally contains custom CUDA implementations of elementwise arithmetic, activations, and SGD parameter updates. The latter uses a dedicated [`sgd_update_kernel`](c/code/src/params.cu):

$$
w_j \leftarrow w_j-\eta\,g_j.
$$

### Computation and differentiation

The implementation connects the individual primitives into trainable models:

- **Tensor representation** with dimensions, device placement, gradients, and backward dependencies.
- **Operations** for arithmetic, matrix multiplication, and activations, with CPU/CUDA implementations.
- **Automatic differentiation** to propagate gradients through the computation graph.
- **Layer and model assembly** using registered weights and biases.
- **Optimization** using CPU updates or a custom CUDA SGD kernel depending on tensor placement.

See [`tensor.h`](c/code/include/tensor.h), [`linear.c`](c/code/src/linear.c), [`model.c`](c/code/src/model.c), and [`params.cu`](c/code/src/params.cu).

## Examples and source

| File | Purpose |
|---|---|
| [`main_network_sin.c`](c/code/examples/main_network_sin.c) | Configurable network trained on a sine target |
| [`main_network_polynomial.c`](c/code/examples/main_network_polynomial.c) | Two-layer `tanh` network and polynomial-target experiment |
| [`main_training_1.c`](c/code/examples/main_training_1.c) | Simple two-layer training experiment |
| [`ops_matmul_cuda.cu`](c/code/src/ops_matmul_cuda.cu) | CUDA matrix multiplication and backward kernels |
| [`params.cu`](c/code/src/params.cu) | Parameter registration, gradient reset, CUDA SGD |

### Building the sine example

The project documents an `nvcc` build in [`Notes.txt`](Notes.txt). From the `c/` directory, compile the sources and the sine example with the NVIDIA CUDA toolkit:

```bash
nvcc code/src/tensor.cu code/src/cuda_utils.cu \
  code/src/ops_add_sub_cpu.c code/src/ops_add_sub_cuda.cu code/src/ops_add_sub.c \
  code/src/ops_mul_div_cpu.c code/src/ops_mul_div_cuda.cu code/src/ops_mul_div.c \
  code/src/ops_matmul.c code/src/ops_matmul_cpu.c code/src/ops_matmul_cuda.cu \
  code/src/linear.c code/src/activation_cpu.c code/src/activation_cuda.cu \
  code/src/activation.c code/src/params.cu code/src/model.c code/src/loss.c \
  code/examples/main_network_sin.c -o main.exe
```

Run from `c/` so that `params.txt` is found at the expected relative path. The training and plotting scripts were developed in a Windows/CUDA workflow; build and plotting paths may require adjustment on other systems.
