# Fused FFN Kernel in Triton

## Architecture

Kernel is implemented in `triton_kernel/fused_kernel.py`

The kernel fuses:
- LayerNorm
- Linear (`x @ W`) 
- GELU activation
- Linear (`x @ W`)

### Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edtallison/fused-ffn-triton/blob/main/notebooks/demo.ipynb)

The notebook will:
- Clone this repo
- Install dependencies (Triton, PyTorch)
- Run the Triton and PyTorch versions of the FFN block
- Benchmark their performance
