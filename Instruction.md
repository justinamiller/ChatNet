# GitHub Copilot Agent Prompt — ChatNet: Pure C# LLM Inference Engine

## Project Identity

You are building **ChatNet**, a high-performance LLM inference library written in **pure C#** with **zero third-party dependencies**. The repository is at `https://github.com/justinamiller/ChatNet`. Clone it, create a feature branch `feature/core-inference-engine`, and do all work there.

---

## Hard Constraints (Non-Negotiable)

These rules apply to **every single line of code** you write. Violating any of these is a build-breaking defect:

### Performance Mandates
- **ZERO third-party NuGet packages.** Only `Microsoft.NET.Sdk` and BCL libraries shipping with .NET 8+.
- **No LINQ anywhere.** No `using System.Linq;`. No `.Where()`, `.Select()`, `.Any()`, `.ToList()`, `.ToArray()` from LINQ. If you need filtering/projection, write explicit `for` loops.
- **No `foreach` loops.** Use indexed `for (int i = 0; i < length; i++)` loops exclusively. `foreach` has enumerator allocation overhead we refuse to pay.
- **No boxing/unboxing.** No casting value types to `object`. Use generics where polymorphism over value types is needed.
- **No closures or lambdas that capture variables** (these allocate hidden classes). If you need a callback, use a static method or a struct-based pattern.
- **No `string` concatenation in hot paths.** Use `Span<char>`, `stackalloc`, or pre-allocated `StringBuilder` where string building is necessary.
- **No `async/await` in the inference hot path.** Synchronous execution only for tensor math and token generation. Async is permitted only at the CLI/IO boundary.
- **Prefer `Span<T>`, `Memory<T>`, `ArrayPool<T>.Shared`, and stack allocation** over heap arrays wherever possible.
- **Use `Vector<T>`, `Vector128<T>`, `Vector256<T>` (System.Numerics / System.Runtime.Intrinsics)** for all tensor math. SIMD is not optional — it is the default.
- **Target .NET 10.0+** to leverage the latest JIT optimizations, AVX2/AVX-512 intrinsics, and `TensorPrimitives`.

### Banned Patterns Checklist
```
❌ using System.Linq;
❌ foreach (var x in collection)
❌ .Where() .Select() .Any() .FirstOrDefault() .ToList() .ToArray()
❌ async/await in math kernels
❌ new List<T>() in hot paths (use ArrayPool or pre-allocated arrays)
❌ string + string in loops
❌ object boxing
❌ Lambda closures that capture local variables
❌ Dictionary<K,V> in hot paths (use flat arrays with computed offsets)
❌ Reflection in runtime paths
```

### Required Patterns Checklist
```
✅ for (int i = 0; i < n; i++)
✅ Span<T> and Memory<T> for slicing
✅ ArrayPool<T>.Shared.Rent() / Return()
✅ stackalloc for small temporary buffers (< 1KB)
✅ Vector256<float> for dot products, matmul, softmax, RMSNorm
✅ MemoryMappedFile for model loading (zero-copy)
✅ BinaryReader with explicit little-endian reads for GGUF parsing
✅ ref struct where appropriate
✅ [MethodImpl(MethodImplOptions.AggressiveInlining)] on small hot methods
✅ readonly struct for immutable data carriers
```

---

## Architecture — Solution Structure

Create a solution with this structure:

```
ChatNet/
├── ChatNet.sln
├── src/
│   ├── ChatNet.Core/                    # The inference library (class library)
│   │   ├── ChatNet.Core.csproj
│   │   ├── Abstractions/
│   │   │   ├── IModel.cs                # Core model interface
│   │   │   ├── ITokenizer.cs            # Tokenizer abstraction
│   │   │   ├── ISampler.cs              # Sampling strategy abstraction
│   │   │   ├── IChatTemplate.cs         # Chat template abstraction
│   │   │   └── IWeightLoader.cs         # Model weight loading abstraction
│   │   ├── Models/
│   │   │   ├── ModelConfig.cs           # Unified model configuration
│   │   │   ├── ModelType.cs             # Enum: Llama, Mistral, Phi, etc.
│   │   │   ├── Llama/
│   │   │   │   ├── LlamaModel.cs        # Llama/TinyLlama inference implementation
│   │   │   │   ├── LlamaWeights.cs      # Weight tensor storage
│   │   │   │   └── LlamaConfig.cs       # Llama-specific hyperparams
│   │   │   └── _Template/
│   │   │       └── README.md            # Template for adding new model architectures
│   │   ├── Tensors/
│   │   │   ├── TensorMath.cs            # SIMD matmul, dot product, softmax, RMSNorm, RoPE
│   │   │   ├── Quantization/
│   │   │   │   ├── DequantQ4_0.cs       # Q4_0 dequantization
│   │   │   │   ├── DequantQ4_1.cs       # Q4_1 dequantization (stub for future)
│   │   │   │   ├── DequantQ8_0.cs       # Q8_0 dequantization (stub for future)
│   │   │   │   └── QuantType.cs         # Enum of quantization types
│   │   │   └── TensorBuffer.cs          # Pooled tensor buffer management
│   │   ├── Tokenizer/
│   │   │   ├── BpeTokenizer.cs          # BPE tokenizer (reads vocab from GGUF)
│   │   │   ├── TokenVocab.cs            # Vocabulary storage
│   │   │   └── SpecialTokens.cs         # BOS, EOS, PAD token handling
│   │   ├── Samplers/
│   │   │   ├── GreedySampler.cs         # Argmax sampling
│   │   │   ├── TemperatureSampler.cs    # Temperature + top-k + top-p
│   │   │   └── SamplerConfig.cs         # Sampling parameters
│   │   ├── Chat/
│   │   │   ├── ChatMessage.cs           # Role + Content message
│   │   │   ├── ChatSession.cs           # Manages conversation context
│   │   │   └── Templates/
│   │   │       ├── LlamaChatTemplate.cs # <|user|> <|assistant|> format
│   │   │       └── ChatMLTemplate.cs    # <|im_start|> format (future)
│   │   ├── Gguf/
│   │   │   ├── GgufReader.cs            # GGUF file parser (header, metadata, tensors)
│   │   │   ├── GgufMetadata.cs          # Metadata key-value storage
│   │   │   ├── GgufTensorInfo.cs        # Tensor descriptor (name, shape, offset, quant type)
│   │   │   └── GgufConstants.cs         # Magic numbers, type enums
│   │   ├── Memory/
│   │   │   ├── PooledBuffer.cs          # ArrayPool wrapper with IDisposable
│   │   │   └── MemoryMappedWeights.cs   # Memory-mapped file access for model weights
│   │   └── InferenceEngine.cs           # Top-level orchestrator: load model → tokenize → infer → sample → decode
│   │
│   └── ChatNet.Cli/                     # Console application for local usage
│       ├── ChatNet.Cli.csproj
│       └── Program.cs                   # CLI entry point with chat REPL
│
└── tests/
    └── ChatNet.Tests/                   # Unit tests (optional, but encouraged)
        ├── ChatNet.Tests.csproj
        └── GgufReaderTests.cs
```

### Project File Requirements

**ChatNet.Core.csproj:**
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <Optimize>true</Optimize>
    <ImplicitUsings>disable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <LangVersion>latest</LangVersion>
  </PropertyGroup>
</Project>
```

**Critical:** Set `ImplicitUsings` to `disable` so LINQ doesn't sneak in. Every `using` statement must be explicit.

**ChatNet.Cli.csproj:**
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <Optimize>true</Optimize>
    <ImplicitUsings>disable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\ChatNet.Core\ChatNet.Core.csproj" />
  </ItemGroup>
</Project>
```

---

## Abstractions — Design for Extensibility

The library must support **multiple model architectures** through clean abstractions. When someone wants to add Mistral, Phi-3, or Gemma support, they implement these interfaces — they never touch the inference engine core.

### IModel
```csharp
public interface IModel : IDisposable
{
    ModelConfig Config { get; }
    
    /// <summary>
    /// Run a forward pass: given input token IDs, produce logits for next token.
    /// </summary>
    /// <param name="tokenIds">Input token sequence</param>
    /// <param name="position">Starting position in the sequence (for KV cache)</param>
    /// <param name="logits">Output buffer for logits (size = vocab_size)</param>
    void Forward(ReadOnlySpan<int> tokenIds, int position, Span<float> logits);
}
```

### ITokenizer
```csharp
public interface ITokenizer
{
    int VocabSize { get; }
    int BosToken { get; }
    int EosToken { get; }
    
    /// <summary>Encode text to token IDs. Caller provides the output buffer.</summary>
    int Encode(ReadOnlySpan<char> text, Span<int> outputTokens);
    
    /// <summary>Decode a single token ID to its string representation.</summary>
    string Decode(int tokenId);
}
```

### ISampler
```csharp
public interface ISampler
{
    /// <summary>Given logits, return the selected token ID.</summary>
    int Sample(ReadOnlySpan<float> logits);
}
```

### IChatTemplate
```csharp
public interface IChatTemplate
{
    /// <summary>Format a conversation into the model's expected prompt format.</summary>
    string FormatPrompt(ReadOnlySpan<ChatMessage> messages);
}
```

---

## GGUF Parsing — Critical Details

The GGUF format is the **only** model format to support initially. You must parse it correctly:

### GGUF File Structure
1. **Magic:** 4 bytes = `0x46475547` ("GGUF" little-endian)
2. **Version:** uint32 (expect version 3)
3. **Tensor count:** uint64
4. **Metadata KV count:** uint64
5. **Metadata KV pairs:** variable length key-value pairs
6. **Tensor infos:** array of tensor descriptors (name, ndims, dims, type, offset)
7. **Alignment padding** to reach the tensor data section
8. **Tensor data:** raw weight bytes at computed offsets

### Metadata Value Types
```
UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6,
BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
```

### Key Metadata Fields to Extract
```
general.architecture          → "llama"
general.name                  → model name
llama.context_length          → max sequence length
llama.embedding_length        → hidden dimension (e.g., 2048)
llama.block_count             → number of transformer layers (e.g., 22)
llama.feed_forward_length     → FFN intermediate size (e.g., 5632)
llama.attention.head_count    → number of attention heads (e.g., 32)
llama.attention.head_count_kv → number of KV heads (for GQA, e.g., 4)
llama.rope.freq_base          → RoPE theta (e.g., 10000.0)
llama.attention.layer_norm_rms_epsilon → RMS norm epsilon
tokenizer.ggml.model          → "llama" (BPE type)
tokenizer.ggml.tokens         → string array of vocab tokens
tokenizer.ggml.scores         → float array of token scores
tokenizer.ggml.token_type     → int array of token types
tokenizer.ggml.bos_token_id   → beginning of sequence token
tokenizer.ggml.eos_token_id   → end of sequence token
```

### Q4_0 Dequantization (Critical for TinyLlama)
Q4_0 stores weights in blocks of 32 values:
- Each block = 2 bytes (float16 scale `d`) + 16 bytes (32 × 4-bit values packed into 16 bytes) = 18 bytes total
- To dequantize block `b`, value `i`:
  ```
  float d = HalfToFloat(block[0..2])     // the scale factor
  byte packed = block[2 + i/2]            // each byte holds 2 values
  int quant = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4)
  float value = (quant - 8) * d           // subtract 8 to center around zero
  ```
- Use `System.Half` or manual float16→float32 conversion for the scale.

### Memory-Mapped Weight Access
```csharp
// Use MemoryMappedFile for zero-copy weight access
var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
// For each tensor, compute its absolute offset = data_section_offset + tensor.offset
// Then read directly from the memory-mapped view
```

---

## Transformer Forward Pass — Llama Architecture

TinyLlama uses the standard Llama 2 architecture. Implement the forward pass exactly:

### Per-Token Forward Pass
```
1. Token Embedding:    x = embedding_table[token_id]     // shape: [dim]
2. For each layer l = 0..n_layers-1:
   a. RMS Norm:        xn = rms_norm(x, attn_norm_weight[l])
   b. QKV Projection:  q = xn @ wq[l],  k = xn @ wk[l],  v = xn @ wv[l]
   c. RoPE:            apply rotary embeddings to q and k at current position
   d. KV Cache:        store k,v into kv_cache[l] at current position
   e. Attention:       for each head, compute scaled dot-product attention over cached K,V
                       attn_out = softmax(Q @ K^T / sqrt(head_dim)) @ V
   f. Output Proj:     attn_result = attn_out @ wo[l]
   g. Residual:        x = x + attn_result
   h. RMS Norm:        xn = rms_norm(x, ffn_norm_weight[l])
   i. FFN (SiLU-gated): 
                       gate = xn @ w1[l]          // gate projection
                       up   = xn @ w3[l]          // up projection
                       hidden = silu(gate) * up    // element-wise
                       ffn_out = hidden @ w2[l]    // down projection
   j. Residual:        x = x + ffn_out
3. Final RMS Norm:     x = rms_norm(x, final_norm_weight)
4. Logits:             logits = x @ output_weight   // shape: [vocab_size]
```

### Key Math Operations to Implement in TensorMath.cs

**All of these MUST use SIMD (Vector256<float> preferred, fallback to Vector128<float>):**

1. **MatVecMul** — Matrix-vector multiply (the hottest path). For Q4_0, dequantize on-the-fly during the multiply.
2. **RMSNorm** — `x[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`
3. **Softmax** — `softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`
4. **RoPE** — Rotary position embeddings. Apply sin/cos rotation to pairs of dimensions.
5. **SiLU** — `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
6. **ElementwiseMul** — `out[i] = a[i] * b[i]`
7. **DotProduct** — `sum(a[i] * b[i])` with SIMD accumulation

### Grouped-Query Attention (GQA)
TinyLlama uses GQA: 32 query heads but only 4 KV heads. Each KV head serves 8 query heads. Your attention implementation must handle this:
```
int kv_head_index = query_head_index / (n_heads / n_kv_heads);
```

### KV Cache
Pre-allocate the KV cache at model load time:
```
float[n_layers, max_seq_len, n_kv_heads, head_dim] for both K and V
```
Use flat arrays with computed strides, not jagged arrays.

---

## GGUF Tensor Name Mapping

TinyLlama GGUF files use these tensor names:
```
token_embd.weight                    → embedding table
blk.{l}.attn_norm.weight             → attention RMS norm weight for layer l
blk.{l}.attn_q.weight                → query projection weight
blk.{l}.attn_k.weight                → key projection weight  
blk.{l}.attn_v.weight                → value projection weight
blk.{l}.attn_output.weight           → attention output projection weight
blk.{l}.ffn_norm.weight              → FFN RMS norm weight
blk.{l}.ffn_gate.weight              → FFN gate (w1)
blk.{l}.ffn_up.weight                → FFN up (w3)
blk.{l}.ffn_down.weight              → FFN down (w2)
output_norm.weight                   → final RMS norm weight
output.weight                        → LM head / output projection
```

---

## BPE Tokenizer Implementation

The tokenizer vocabulary comes from the GGUF metadata. Implement SentencePiece-style BPE:

1. **Load vocabulary** from `tokenizer.ggml.tokens` and `tokenizer.ggml.scores`
2. **Encode algorithm:**
   - Convert input text to UTF-8 bytes
   - Initialize each byte as a separate token (use the byte fallback tokens `<0xHH>`)
   - Repeatedly find the highest-scoring adjacent pair that exists in vocab
   - Merge that pair into a single token
   - Repeat until no more merges are possible
3. **Prepend BOS token** if the model requires it (TinyLlama does)
4. **Handle the `▁` (U+2581) prefix** that SentencePiece uses for space tokens

Use pre-built lookup structures (sorted arrays or hash tables built at load time) for fast pair lookup. No Dictionary in the merge loop — use a flat array-based approach.

---

## TinyLlama Chat Template

TinyLlama 1.1B Chat v1.0 uses this template:
```
<|system|>
You are a helpful assistant.</s>
<|user|>
{user_message}</s>
<|assistant|>
```

Implement this in `LlamaChatTemplate.cs`. The `</s>` is the EOS token in text form.

---

## CLI Application (ChatNet.Cli)

Build an interactive chat REPL:

```
ChatNet v0.1.0 — Pure C# LLM Inference
Loading model: /Users/justinmiller/Downloads/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
Model: TinyLlama 1.1B Chat v1.0 | Params: 1.1B | Quant: Q4_0
Loaded in 245ms | Memory: 638 MB

> What is the capital of France?

The capital of France is Paris. Paris has been the capital...

[132 tokens | 18.4 tok/s | 7.2s]

> 
```

### CLI Requirements:
- Accept model path as command-line argument: `dotnet run -- --model /path/to/model.gguf`
- Show loading statistics (time, memory)
- Interactive chat loop with `>` prompt
- Show generation statistics after each response (tokens, speed)
- Support `Ctrl+C` to cancel generation, `exit` or `quit` to close
- Stream tokens to console as they're generated (print each token immediately)

---

## Acceptance Criteria — Definition of Done

The implementation is **complete** when ALL of the following pass:

### Functional Requirements
- [ ] `dotnet build` succeeds with zero warnings on the entire solution
- [ ] `dotnet run --project src/ChatNet.Cli -- --model /Users/justinmiller/Downloads/tinyllama-1.1b-chat-v1.0.Q4_0.gguf` loads the model successfully
- [ ] Asking "What is the capital of France?" produces a coherent response that includes "Paris"
- [ ] The response is generated token-by-token with streaming output to console
- [ ] The chat session maintains context for follow-up questions
- [ ] Typing `exit` cleanly shuts down

### Quality Requirements
- [ ] `grep -r "using System.Linq" src/` returns ZERO results
- [ ] `grep -r "foreach" src/` returns ZERO results (excluding comments/strings)
- [ ] Zero third-party PackageReference entries in any .csproj
- [ ] All tensor math uses System.Numerics.Vector or System.Runtime.Intrinsics
- [ ] Memory-mapped file used for weight loading
- [ ] ArrayPool used for temporary buffers in inference
- [ ] No allocations in the per-token generation hot path (verified by reviewing code)

### Performance Targets (on modern x86_64)
- Model load time: < 2 seconds
- Token generation: > 10 tokens/second on CPU
- Memory usage: < 1 GB for Q4_0 TinyLlama (model is ~638MB)

---

## Implementation Order — Build Incrementally

Follow this exact order. **Test each step before moving on:**

### Phase 1: Foundation
1. Create the solution structure and project files
2. Implement `GgufReader` — parse the GGUF header, metadata, and tensor info
3. Write a test that loads the TinyLlama file and prints all metadata keys and tensor names
4. Verify: tensor count, layer count, vocab size, dimension sizes all match expected values

### Phase 2: Tokenizer
5. Implement `BpeTokenizer` — load vocab from GGUF metadata, implement encode/decode
6. Test: encode "What is the capital of France?" and print token IDs
7. Test: decode each token back and verify round-trip

### Phase 3: Tensor Math
8. Implement SIMD `DotProduct`, `MatVecMul`, `RMSNorm`, `Softmax`, `SiLU`, `RoPE`
9. Implement `DequantQ4_0` — dequantize Q4_0 blocks to float32
10. Implement `MatVecMulQ4_0` — fused dequant + matvec for maximum performance

### Phase 4: Model Loading
11. Implement `LlamaWeights` — memory-map the file and resolve tensor pointers
12. Implement `LlamaModel.Forward()` — the full transformer forward pass
13. Test: run a single forward pass on BOS token, verify logits are non-zero and finite

### Phase 5: Generation
14. Implement `GreedySampler` (argmax)
15. Implement `InferenceEngine` — the generate loop (prompt encode → forward → sample → decode → repeat)
16. Test: generate 10 tokens from a simple prompt, verify they're real words

### Phase 6: Chat
17. Implement `LlamaChatTemplate` and `ChatSession`
18. Build the CLI REPL
19. **Final test: "What is the capital of France?" → response includes "Paris"**

### Phase 7: Polish
20. Add `TemperatureSampler` with top-k and top-p
21. Add generation parameters (max tokens, temperature, etc.) to CLI flags
22. Profile and optimize any bottlenecks

---

## Debugging Checklist — When Output is Garbage

If the model produces gibberish, check these **in order**:

1. **GGUF alignment:** Tensor data offset must account for alignment padding (typically 32-byte aligned). `data_start = metadata_end rounded up to alignment boundary`.
2. **Q4_0 dequant:** Verify the half-float scale conversion. Test against known values. The subtraction is `(quant - 8)`, not `(quant - 7)`.
3. **Tensor dimensions:** Weight matrices in GGUF are stored as `[out_features, in_features]`. MatVecMul computes `output[i] = dot(weights[i, :], input)`.
4. **RoPE frequency:** Must use the correct `rope_freq_base` from metadata (10000.0 for TinyLlama). Apply RoPE to Q and K only, not V.
5. **GQA head mapping:** Verify each query head maps to the correct KV head: `kv_head = q_head / (n_heads / n_kv_heads)`.
6. **Attention scale:** Divide by `sqrt(head_dim)`, where `head_dim = dim / n_heads`.
7. **Attention mask:** For position `pos`, only attend to positions `0..pos` (causal mask). Set future positions to `-infinity` before softmax.
8. **BOS token:** TinyLlama requires BOS (token ID 1) prepended to every prompt.
9. **Byte order:** GGUF is always little-endian. Ensure your reads match.
10. **RMSNorm:** The formula is `x * w / sqrt(mean(x²) + eps)` — don't forget the weight multiplication.

---

## Reference: TinyLlama 1.1B Expected Values

Use these to validate your GGUF parsing:
```
Architecture:        llama
Vocab Size:          32000
Embedding Dim:       2048
Layers:              22
Attention Heads:     32
KV Heads:            4
Head Dim:            64  (= 2048 / 32)
FFN Hidden Dim:      5632
Context Length:      2048
RoPE Theta:          10000.0
RMS Norm Epsilon:    1e-5
Quantization:        Q4_0 (mostly, some tensors may be F32)
```

---

## Final Notes

- **Do not hallucinate APIs.** If you're unsure whether a .NET BCL class exists, verify before using it. Stick to `System.Numerics`, `System.Runtime.Intrinsics`, `System.Buffers`, `System.IO.MemoryMappedFiles`, `System.IO`, `System.Text`, `System.Collections.Generic`, `System.Runtime.CompilerServices`, `System.Runtime.InteropServices`.
- **Comment the math.** Every tensor operation should have a one-line comment explaining what it's computing and the expected shapes.
- **Name things clearly.** `attnQueryWeight` not `wq`. `feedForwardGate` not `w1`. This is a library people will read.
- **Keep going until the acceptance criteria pass.** Don't stop at "here's the structure" — implement every method, debug every issue, until "What is the capital of France?" returns "Paris".
