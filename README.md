# ChatNet

A high-performance LLM inference engine written in **pure C#** with **zero third-party dependencies**. Runs GGUF models locally using .NET 10+.

## Features

- **Pure C#** — No NuGet packages, no native libraries. Only BCL/System.* APIs.
- **GGUF Support** — Full GGUF v3 parser with Q4_0 quantization (Q4_1, Q8_0 stubs ready).
- **SIMD Accelerated** — Vector<T> for dot products, matmul, RMSNorm, softmax.
- **Memory Mapped** — Zero-copy model weight access via MemoryMappedFile.
- **Zero Allocation Token Loop** — ArrayPool-backed buffers, pre-allocated scratch tensors.
- **Llama Architecture** — TinyLlama, Llama 2, with GQA support.
- **BPE Tokenizer** — SentencePiece-style tokenizer loaded from GGUF metadata.
- **Streaming** — Token-by-token output during generation.

## Quick Start

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download) or later
- A GGUF model file (e.g., TinyLlama 1.1B Chat Q4_0)

### Run a single prompt

```bash
dotnet run --project src/ChatNet.Cli -- \
  --model "/path/to/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" \
  --prompt "What is the capital of France?" \
  --max-tokens 128 \
  --temp 0.0
```

### Interactive chat mode

```bash
dotnet run --project src/ChatNet.Cli -- \
  --model "/path/to/tinyllama-1.1b-chat-v1.0.gguf" \
  --interactive
```

### Build the solution

```bash
dotnet build ChatNet.sln
```

## CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | Path to GGUF model file | (required) |
| `--prompt` | `-p` | Input prompt text | (interactive) |
| `--max-tokens` | `-n` | Maximum tokens to generate | 128 |
| `--temp` | `-t` | Sampling temperature (0.0 = greedy) | 0.0 |
| `--top-k` | | Top-K sampling | 40 |
| `--top-p` | | Top-P / nucleus sampling | 0.9 |
| `--interactive` | `-i` | Enter interactive chat mode | false |
| `--help` | `-h` | Show help | |

## Project Structure

```
ChatNet/
├── ChatNet.sln
├── src/
│   ├── ChatNet.Core/           # Core inference library
│   │   ├── Abstractions/       # IModel, ITokenizer, ISampler, IChatTemplate, IWeightLoader
│   │   ├── Gguf/               # GGUF file parser and metadata
│   │   ├── Memory/             # PooledBuffer, MemoryMappedWeights
│   │   ├── Tensors/            # TensorMath (SIMD), Quantization (Q4_0, Q4_1, Q8_0)
│   │   ├── Tokenizer/          # BPE tokenizer from GGUF vocab
│   │   ├── Models/Llama/       # Llama model forward pass, weights, config
│   │   ├── Samplers/           # Greedy and temperature samplers
│   │   ├── Chat/               # Chat session, templates (Llama, ChatML)
│   │   └── InferenceEngine.cs  # Top-level orchestrator
│   └── ChatNet.Cli/            # Console application
│       └── Program.cs          # CLI entry point
└── tests/
    └── ChatNet.Tests/          # Unit tests
```

## Architecture

The core runtime is a reusable class library (`ChatNet.Core`) designed for extensibility:

- **IModel** — Forward pass abstraction. Implement for new architectures.
- **ITokenizer** — Encode/decode with Span-based APIs.
- **ISampler** — Pluggable sampling strategies (greedy, temperature+top-k+top-p).
- **IChatTemplate** — Chat prompt formatting (Llama/Zephyr, ChatML, etc.).
- **IWeightLoader** — Model weight access abstraction (GGUF first, extensible).

## Performance Constraints

This project enforces strict performance rules:

- No LINQ, no foreach, no closures in hot paths
- No string concatenation in loops
- SIMD (System.Numerics.Vector) for all tensor math
- ArrayPool for temporary buffers
- MemoryMappedFile for weight access
- Pre-allocated scratch buffers for the forward pass

## Supported Models

| Model | Status |
|-------|--------|
| TinyLlama 1.1B Chat Q4_0 | Supported |
| Llama 2 (Q4_0) | Architecture supported |
| Other Llama-family | Architecture supported |

## License

See repository license.
