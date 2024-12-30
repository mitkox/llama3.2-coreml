# LLaMA 3.2 CoreML

This repository contains the implementation for running Meta's LLaMA 3.2 model on Apple Silicon using Core ML. It provides tools for exporting, quantizing, and running the LLaMA model with optimized key-value caching for improved performance.

## Features

- Export LLaMA 3.2 model to Core ML format
- Optimized key-value caching implementation
- INT4 quantization support for reduced model size
- Batch processing capabilities
- Configurable context window size

## Prerequisites

- Python 3.8+
- Apple Silicon Mac (M1/M2/M3)
- PyTorch
- Core ML Tools
- Hugging Face Transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mitkox/llama3.2-coreml.git
cd llama3.2-coreml
```

2. Install the required dependencies:
```bash
pip install torch coremltools transformers
```

## Usage

### Exporting the Model

To export the LLaMA model to Core ML format with optional quantization:

```bash
python run_model.py --output-dir /path/to/output
```

Options:
- `--output-dir`: Directory to save the exported models (default: current directory)
- `--skip-quantization`: Skip the model quantization step

### Model Architecture

The implementation includes three main components:

1. `llama_implementation.py`: Contains the base LLaMA model implementation
2. `kv_cache.py`: Implements optimized key-value caching
3. `export_model.py`: Handles model export and quantization

## Model Outputs

The export process generates two files:
- `llama_3_2_3b.mlpackage`: Base model in Core ML format
- `llama_3_2_3b_int4.mlpackage`: Quantized model (INT4) for reduced size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for the LLaMA model
- Apple's Core ML team for the ML framework
- Hugging Face for the Transformers library
