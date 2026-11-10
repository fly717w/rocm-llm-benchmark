# ROCm LLM Benchmark

A small Python project for measuring local language-model inference with PyTorch on AMD ROCm. The planned AMD Developer Cloud experiment uses Qwen2.5-0.5B-Instruct to verify that a model can load and generate text on an AMD GPU, then measures generation latency, throughput, and peak GPU tensor memory.

**Status:** The benchmark script is ready for an initial run. AMD GPU compatibility and performance are pending hardware access. This repository contains no measured GPU results.

## Setup

1. Use a Linux environment with a supported AMD GPU and ROCm-enabled PyTorch. If your cloud image already includes it, keep that build. Otherwise, follow the [AMD PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html) for your GPU and ROCm version.
2. In that same Python environment, check the HIP build and install Transformers:

   ```bash
   python -c "import torch; print('HIP:', torch.version.hip, '| GPU visible:', torch.cuda.is_available())"
   python -m pip install -r requirements.txt
   ```

   On ROCm, expect a HIP version and `GPU visible: True`. PyTorch exposes AMD GPU operations through the `torch.cuda` namespace.

3. Run the benchmark. The first run downloads model files from Hugging Face.

   ```bash
   python benchmark.py --device rocm --output results/rocm.json
   ```

   To check the model-loading and reporting workflow on a machine with CPU PyTorch:

   ```bash
   python benchmark.py --device cpu --warmup 1 --runs 2 --max-new-tokens 8
   ```

## What the report measures

- **Latency:** Median (p50) and nearest-rank p95 of whole generation calls. Timing excludes model loading, tokenization, device transfer of inputs, and warm-up runs.
- **Throughput:** Actual generated tokens divided by total timed generation seconds. The model may finish before reaching the token limit.
- **GPU memory:** Peak memory allocated for PyTorch tensors during the measured runs, including model weights. This differs from whole-device memory usage.

Each report includes raw samples, model and environment details, and benchmark settings. CPU mode uses FP32 and ROCm mode uses FP16. Treat the CPU command as a smoke test.

## Planned cloud experiment

1. Record the GPU model, HIP and PyTorch versions, and the model commit. Pass the commit with `--revision` for reproducibility.
2. Run the default prompt at 32 and 64 maximum new tokens, with two warm-up runs and 20 measured runs per setting. Save separate JSON reports with `--output`.
3. Review actual token counts alongside latency, throughput, and peak allocated memory. Add the measured reports and a short summary after completing the experiment.

Default model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Model weights are downloaded separately.
