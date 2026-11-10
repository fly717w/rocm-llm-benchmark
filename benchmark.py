"""Measure single-request text generation latency on CPU or AMD ROCm."""

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "Explain the difference between a CPU and a GPU in two sentences."


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def summarize(samples):
    latencies = sorted(sample["latency_ms"] for sample in samples)
    total_tokens = sum(sample["generated_tokens"] for sample in samples)
    return {
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "generated_tokens_per_second": 1000 * total_tokens / sum(latencies),
    }


def benchmark(args):
    import torch

    if args.device == "rocm" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("ROCm requires a HIP-enabled PyTorch build and a visible AMD GPU.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_gpu = args.device == "rocm"
    # PyTorch uses the cuda device namespace for HIP.
    device = torch.device("cuda:0" if use_gpu else "cpu")
    dtype = torch.float16 if use_gpu else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype
    ).to(device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    samples = []
    with torch.inference_mode():
        for run in range(args.warmup + args.runs):
            measured = run >= args.warmup
            if use_gpu:
                torch.cuda.synchronize(device)
                if measured and run == args.warmup:
                    torch.cuda.reset_peak_memory_stats(device)

            started = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            if use_gpu:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000

            if measured:
                samples.append({
                    "latency_ms": elapsed_ms,
                    "generated_tokens": output.shape[-1] - prompt_tokens,
                })
            del output

    peak_bytes = torch.cuda.max_memory_allocated(device) if use_gpu else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision_requested": args.revision,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device) if use_gpu else None,
        "pytorch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "dtype": str(dtype),
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "peak_gpu_allocated_mib": peak_bytes / 1024**2 if peak_bytes is not None else None,
        "summary": summarize(samples),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["rocm", "cpu"], default="rocm")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model with a chat template")
    parser.add_argument("--revision", default="main", help="Model tag or commit SHA")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=positive_int, default=32)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--runs", type=positive_int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    try:
        report = json.dumps(benchmark(args), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")

    print(report)


if __name__ == "__main__":
    main()
