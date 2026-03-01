using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;
using ChatNet.Core.Tensors;
using ChatNet.Core.Tensors.Quantization;

namespace ChatNet.Core.Models.Llama
{
    /// <summary>
    /// Llama-family model implementation (supports TinyLlama, Llama 2, etc.)
    /// Implements the full transformer forward pass with KV cache.
    /// </summary>
    public sealed class LlamaModel : IModel
    {
        private readonly LlamaConfig _cfg;
        private readonly LlamaWeights _weights;
        private readonly ModelConfig _modelConfig;

        // KV cache: flat arrays [layer * maxSeq * kvDim]
        private readonly float[] _keyCache;
        private readonly float[] _valueCache;

        // Pre-allocated scratch buffers for forward pass (no per-token allocations)
        private readonly float[] _x;         // current hidden state [dim]
        private readonly float[] _xNorm;     // after RMS norm [dim]
        private readonly float[] _q;         // query [dim] = nHeads * headDim
        private readonly float[] _k;         // key [kvDim] = nKvHeads * headDim
        private readonly float[] _v;         // value [kvDim]
        private readonly float[] _attnOut;   // attention output [dim]
        private readonly float[] _gate;      // FFN gate [hiddenDim]
        private readonly float[] _up;        // FFN up [hiddenDim]
        private readonly float[] _ffnOut;    // FFN down output [dim]
        private readonly float[] _attnScores; // attention scores [maxSeqLen] per head

        // Reusable single-element buffer for generation (avoids ReadOnlySpan<int>(ref local) issues)
        private readonly int[] _singleTokenBuf = new int[1];

        public ModelConfig Config => _modelConfig;

        /// <summary>Debug flag: when true, prints diagnostic information to stderr.</summary>
        public static bool DebugEnabled { get; set; }

        public LlamaModel(ModelConfig modelConfig, LlamaWeights weights)
        {
            _modelConfig = modelConfig;
            _cfg = new LlamaConfig(modelConfig);
            _weights = weights;

            int maxSeq = _cfg.ContextLength;
            int kvDim = _cfg.KvDim;
            int dim = _cfg.Dim;
            int hiddenDim = _cfg.HiddenDim;

            // Allocate KV cache
            _keyCache = new float[_cfg.LayerCount * maxSeq * kvDim];
            _valueCache = new float[_cfg.LayerCount * maxSeq * kvDim];

            // Allocate scratch buffers
            _x = new float[dim];
            _xNorm = new float[dim];
            _q = new float[dim];
            _k = new float[kvDim];
            _v = new float[kvDim];
            _attnOut = new float[dim];
            _gate = new float[hiddenDim];
            _up = new float[hiddenDim];
            _ffnOut = new float[dim];
            _attnScores = new float[maxSeq];

            if (DebugEnabled)
            {
                Console.Error.WriteLine("[DEBUG] LlamaModel constructed:");
                Console.Error.WriteLine("[DEBUG]   dim=" + dim + " layers=" + _cfg.LayerCount +
                    " heads=" + _cfg.HeadCount + "/" + _cfg.KvHeadCount +
                    " headDim=" + _cfg.HeadDim + " kvDim=" + kvDim +
                    " ffn=" + hiddenDim + " vocab=" + _cfg.VocabSize);
                Console.Error.WriteLine("[DEBUG]   embType=" + _weights.EmbeddingType +
                    " outType=" + _weights.OutputType);
                Console.Error.WriteLine("[DEBUG]   attnQ[0]=" + _weights.AttnQType[0] +
                    " attnK[0]=" + _weights.AttnKType[0] +
                    " ffnGate[0]=" + _weights.FfnGateType[0]);
            }
        }

        /// <summary>
        /// Run forward pass for a single token at the given position.
        /// </summary>
        public void Forward(ReadOnlySpan<int> tokenIds, int position, Span<float> logits)
        {
            int dim = _cfg.Dim;
            int kvDim = _cfg.KvDim;
            int headDim = _cfg.HeadDim;
            int nHeads = _cfg.HeadCount;
            int nKvHeads = _cfg.KvHeadCount;
            int kvMul = _cfg.KvMul;
            int layers = _cfg.LayerCount;
            int hiddenDim = _cfg.HiddenDim;
            int vocabSize = _cfg.VocabSize;

            // Process each token in the sequence
            for (int t = 0; t < tokenIds.Length; t++)
            {
                int tokenId = tokenIds[t];
                int pos = position + t;

                // Step 1: Token embedding
                LoadEmbedding(tokenId, _x.AsSpan(0, dim));

                if (DebugEnabled && pos == 0)
                {
                    float embSum = 0f;
                    for (int ei = 0; ei < dim; ei++) embSum += _x[ei] * _x[ei];
                    Console.Error.WriteLine("[DEBUG] Embedding[token=" + tokenId + "] L2=" +
                        MathF.Sqrt(embSum).ToString("F6") +
                        " first5=[" + _x[0].ToString("F4") + "," + _x[1].ToString("F4") +
                        "," + _x[2].ToString("F4") + "," + _x[3].ToString("F4") +
                        "," + _x[4].ToString("F4") + "]");
                }

                // Step 2: Process each transformer layer
                for (int l = 0; l < layers; l++)
                {
                    // 2a: RMS Norm before attention
                    ReadOnlySpan<float> attnNormW = GetF32Weights(_weights.GetAttnNormWeight(l), dim);
                    TensorMath.RmsNorm(_x.AsSpan(0, dim), attnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    // 2b: QKV projections
                    unsafe
                    {
                        MatVecMulByType(_weights.GetAttnQWeight(l), _weights.AttnQType[l],
                            _xNorm.AsSpan(0, dim), _q.AsSpan(0, dim), dim, dim);

                        MatVecMulByType(_weights.GetAttnKWeight(l), _weights.AttnKType[l],
                            _xNorm.AsSpan(0, dim), _k.AsSpan(0, kvDim), kvDim, dim);

                        MatVecMulByType(_weights.GetAttnVWeight(l), _weights.AttnVType[l],
                            _xNorm.AsSpan(0, dim), _v.AsSpan(0, kvDim), kvDim, dim);
                    }

                    // 2c: Apply RoPE to Q and K
                    TensorMath.ApplyRoPE(_q.AsSpan(0, dim), _k.AsSpan(0, kvDim),
                        pos, headDim, nHeads, nKvHeads, _cfg.RopeFreqBase);

                    // 2d: Store K and V into KV cache
                    int kvCacheLayerOffset = l * _cfg.ContextLength * kvDim;
                    int kvCachePos = kvCacheLayerOffset + pos * kvDim;
                    Array.Copy(_k, 0, _keyCache, kvCachePos, kvDim);
                    Array.Copy(_v, 0, _valueCache, kvCachePos, kvDim);

                    // 2e: Multi-head attention with GQA
                    ComputeAttention(l, pos, headDim, nHeads, nKvHeads, kvMul, kvDim, dim);

                    // 2f: Output projection: attnResult = attnOut @ Wo
                    unsafe
                    {
                        MatVecMulByType(_weights.GetAttnOutputWeight(l), _weights.AttnOutputType[l],
                            _attnOut.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim, dim);
                    }

                    // 2g: Residual connection
                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);

                    // 2h: RMS Norm before FFN
                    ReadOnlySpan<float> ffnNormW = GetF32Weights(_weights.GetFfnNormWeight(l), dim);
                    TensorMath.RmsNorm(_x.AsSpan(0, dim), ffnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    // 2i: FFN (SiLU-gated)
                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnGateWeight(l), _weights.FfnGateType[l],
                            _xNorm.AsSpan(0, dim), _gate.AsSpan(0, hiddenDim), hiddenDim, dim);

                        MatVecMulByType(_weights.GetFfnUpWeight(l), _weights.FfnUpType[l],
                            _xNorm.AsSpan(0, dim), _up.AsSpan(0, hiddenDim), hiddenDim, dim);
                    }

                    TensorMath.SiluElementwiseMul(_gate.AsSpan(0, hiddenDim), _up.AsSpan(0, hiddenDim), hiddenDim);

                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnDownWeight(l), _weights.FfnDownType[l],
                            _gate.AsSpan(0, hiddenDim), _ffnOut.AsSpan(0, dim), dim, hiddenDim);
                    }

                    // 2j: Residual connection
                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);

                    if (DebugEnabled && pos == 0 && (l == 0 || l == layers - 1))
                    {
                        float xSum = 0f;
                        for (int xi = 0; xi < dim; xi++) xSum += _x[xi] * _x[xi];
                        Console.Error.WriteLine("[DEBUG] After layer " + l + ": x L2=" +
                            MathF.Sqrt(xSum).ToString("F6"));
                    }
                }

                // Step 3: Final RMS Norm
                ReadOnlySpan<float> finalNormW = GetF32Weights(_weights.GetFinalNormWeight(), dim);
                TensorMath.RmsNorm(_x.AsSpan(0, dim), finalNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                // Step 4: LM head - compute logits (only for the last token)
                if (t == tokenIds.Length - 1)
                {
                    unsafe
                    {
                        MatVecMulByType(_weights.GetOutputWeight(), _weights.OutputType,
                            _xNorm.AsSpan(0, dim), logits, vocabSize, dim);
                    }

                    if (DebugEnabled && pos <= 1)
                    {
                        // Print logit statistics
                        float lMin = logits[0], lMax = logits[0], lSum = 0f;
                        int nanCount = 0;
                        for (int li = 0; li < vocabSize; li++)
                        {
                            float v = logits[li];
                            if (float.IsNaN(v)) { nanCount++; continue; }
                            if (v < lMin) lMin = v;
                            if (v > lMax) lMax = v;
                            lSum += v;
                        }
                        Console.Error.WriteLine("[DEBUG] Logits[pos=" + pos + "]: min=" +
                            lMin.ToString("F4") + " max=" + lMax.ToString("F4") +
                            " mean=" + (lSum / vocabSize).ToString("F4") +
                            " NaN=" + nanCount +
                            " logits[0]=" + logits[0].ToString("F4") +
                            " logits[1]=" + logits[1].ToString("F4") +
                            " logits[2]=" + logits[2].ToString("F4"));
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private unsafe void ComputeAttention(int layer, int pos, int headDim, int nHeads, int nKvHeads, int kvMul, int kvDim, int dim)
        {
            int maxSeq = _cfg.ContextLength;
            int kvCacheLayerOffset = layer * maxSeq * kvDim;
            int seqLen = pos + 1;

            // Hoist scale computation outside the head loop (headDim is constant across heads)
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Pin all arrays once to eliminate repeated pinning and enable pointer arithmetic
            // This removes all bounds checks from the inner loops
            fixed (float* pQ = _q)
            fixed (float* pKeyCache = _keyCache)
            fixed (float* pValueCache = _valueCache)
            fixed (float* pAttnScores = _attnScores)
            fixed (float* pAttnOut = _attnOut)
            {
                // Clear attention output via pointer
                int dimBytes = dim * sizeof(float);
                new Span<float>(pAttnOut, dim).Clear();

                for (int h = 0; h < nHeads; h++)
                {
                    int qOffset = h * headDim;
                    int kvHead = h / kvMul;
                    int kvOffset = kvHead * headDim;

                    float* qPtr = pQ + qOffset;

                    // --- QK^T: compute attention scores ---
                    // For each position, dot(Q[h], K_cache[layer, pos, kvHead])
                    for (int p = 0; p < seqLen; p++)
                    {
                        float* kPtr = pKeyCache + kvCacheLayerOffset + p * kvDim + kvOffset;

                        // SIMD dot product with dual accumulators for ILP
                        float score = 0f;
                        int d = 0;

                        if (Vector.IsHardwareAccelerated && headDim >= Vector<float>.Count)
                        {
                            int vecLen = Vector<float>.Count;
                            var vAcc0 = Vector<float>.Zero;
                            var vAcc1 = Vector<float>.Zero;
                            int limit = headDim - 2 * vecLen + 1;

                            // Dual-accumulator unrolled loop
                            for (; d < limit; d += 2 * vecLen)
                            {
                                vAcc0 += *(Vector<float>*)(qPtr + d) * *(Vector<float>*)(kPtr + d);
                                vAcc1 += *(Vector<float>*)(qPtr + d + vecLen) * *(Vector<float>*)(kPtr + d + vecLen);
                            }

                            // Single-vector remainder
                            int singleLimit = headDim - vecLen + 1;
                            for (; d < singleLimit; d += vecLen)
                            {
                                vAcc0 += *(Vector<float>*)(qPtr + d) * *(Vector<float>*)(kPtr + d);
                            }

                            score = VectorSumFast(vAcc0 + vAcc1);
                        }

                        // Scalar tail
                        for (; d < headDim; d++)
                        {
                            score += qPtr[d] * kPtr[d];
                        }

                        pAttnScores[p] = score * scale;
                    }

                    // Softmax over scores[0..seqLen-1]
                    TensorMath.Softmax(new Span<float>(pAttnScores, seqLen), seqLen);

                    // --- AV: weighted sum of value vectors ---
                    float* outPtr = pAttnOut + qOffset;

                    for (int p = 0; p < seqLen; p++)
                    {
                        float attnWeight = pAttnScores[p];
                        if (attnWeight == 0f) continue;

                        float* vPtr = pValueCache + kvCacheLayerOffset + p * kvDim + kvOffset;

                        int d = 0;
                        if (Vector.IsHardwareAccelerated && headDim >= Vector<float>.Count)
                        {
                            int vecLen = Vector<float>.Count;
                            var vWeight = new Vector<float>(attnWeight);
                            int limit = headDim - vecLen + 1;
                            for (; d < limit; d += vecLen)
                            {
                                *(Vector<float>*)(outPtr + d) += *(Vector<float>*)(vPtr + d) * vWeight;
                            }
                        }

                        // Scalar tail
                        for (; d < headDim; d++)
                        {
                            outPtr[d] += attnWeight * vPtr[d];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Fast horizontal vector sum using pairwise add when available.
        /// Falls back to element-wise loop otherwise.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float VectorSumFast(Vector<float> v)
        {
            // For Vector<float>, the JIT produces efficient hadd sequences on x86
            float sum = 0f;
            for (int i = 0; i < Vector<float>.Count; i++)
            {
                sum += v[i];
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LoadEmbedding(int tokenId, Span<float> output)
        {
            int dim = _cfg.Dim;
            ReadOnlySpan<byte> embData = _weights.GetEmbeddingData();

            if (_weights.EmbeddingType == GgmlType.F32)
            {
                ReadOnlySpan<float> allEmb = MemoryMarshal.Cast<byte, float>(embData);
                int offset = tokenId * dim;
                allEmb.Slice(offset, dim).CopyTo(output);
            }
            else if (_weights.EmbeddingType == GgmlType.F16)
            {
                int offset = tokenId * dim * 2;
                for (int i = 0; i < dim; i++)
                {
                    output[i] = DequantQ4_0.HalfToFloat(embData[offset + i * 2], embData[offset + i * 2 + 1]);
                }
            }
            else
            {
                // Generic dequantization path for all quantized types
                GetDequantInfo(_weights.EmbeddingType, out int blockSize, out int bytesPerBlock,
                    out DequantizeDelegate dequant);
                int blocksPerRow = dim / blockSize;
                int bytesPerRow = blocksPerRow * bytesPerBlock;
                int rowOffset = tokenId * bytesPerRow;
                dequant(embData.Slice(rowOffset, bytesPerRow), output, dim);
            }
        }

        private delegate void DequantizeDelegate(ReadOnlySpan<byte> data, Span<float> output, int count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void GetDequantInfo(GgmlType type, out int blockSize, out int bytesPerBlock,
            out DequantizeDelegate dequant)
        {
            switch (type)
            {
                case GgmlType.Q4_0:
                    blockSize = DequantQ4_0.BlockSize; bytesPerBlock = DequantQ4_0.BytesPerBlock;
                    dequant = DequantQ4_0.Dequantize; return;
                case GgmlType.Q4_1:
                    blockSize = DequantQ4_1.BlockSize; bytesPerBlock = DequantQ4_1.BytesPerBlock;
                    dequant = DequantQ4_1.Dequantize; return;
                case GgmlType.Q5_0:
                    blockSize = DequantQ5_0.BlockSize; bytesPerBlock = DequantQ5_0.BytesPerBlock;
                    dequant = DequantQ5_0.Dequantize; return;
                case GgmlType.Q5_1:
                    blockSize = DequantQ5_1.BlockSize; bytesPerBlock = DequantQ5_1.BytesPerBlock;
                    dequant = DequantQ5_1.Dequantize; return;
                case GgmlType.Q8_0:
                    blockSize = DequantQ8_0.BlockSize; bytesPerBlock = DequantQ8_0.BytesPerBlock;
                    dequant = DequantQ8_0.Dequantize; return;
                case GgmlType.Q2K:
                    blockSize = DequantQ2K.BlockSize; bytesPerBlock = DequantQ2K.BytesPerBlock;
                    dequant = DequantQ2K.Dequantize; return;
                case GgmlType.Q3K:
                    blockSize = DequantQ3K.BlockSize; bytesPerBlock = DequantQ3K.BytesPerBlock;
                    dequant = DequantQ3K.Dequantize; return;
                case GgmlType.Q4K:
                    blockSize = DequantQ4K.BlockSize; bytesPerBlock = DequantQ4K.BytesPerBlock;
                    dequant = DequantQ4K.Dequantize; return;
                case GgmlType.Q5K:
                    blockSize = DequantQ5K.BlockSize; bytesPerBlock = DequantQ5K.BytesPerBlock;
                    dequant = DequantQ5K.Dequantize; return;
                case GgmlType.Q6K:
                    blockSize = DequantQ6K.BlockSize; bytesPerBlock = DequantQ6K.BytesPerBlock;
                    dequant = DequantQ6K.Dequantize; return;
                case GgmlType.Q8K:
                    blockSize = DequantQ8K.BlockSize; bytesPerBlock = DequantQ8K.BytesPerBlock;
                    dequant = DequantQ8K.Dequantize; return;
                case GgmlType.IQ4NL:
                    blockSize = DequantIQ.IQ4NL.BlockSize; bytesPerBlock = DequantIQ.IQ4NL.BytesPerBlock;
                    dequant = DequantIQ.IQ4NL.Dequantize; return;
                case GgmlType.IQ4XS:
                    blockSize = DequantIQ.IQ4XS.BlockSize; bytesPerBlock = DequantIQ.IQ4XS.BytesPerBlock;
                    dequant = DequantIQ.IQ4XS.Dequantize; return;
                case GgmlType.IQ3S:
                    blockSize = DequantIQ.IQ3S.BlockSize; bytesPerBlock = DequantIQ.IQ3S.BytesPerBlock;
                    dequant = DequantIQ.IQ3S.Dequantize; return;
                case GgmlType.IQ3XXS:
                    blockSize = DequantIQ.IQ3XXS.BlockSize; bytesPerBlock = DequantIQ.IQ3XXS.BytesPerBlock;
                    dequant = DequantIQ.IQ3XXS.Dequantize; return;
                case GgmlType.IQ2XS:
                    blockSize = DequantIQ.IQ2XS.BlockSize; bytesPerBlock = DequantIQ.IQ2XS.BytesPerBlock;
                    dequant = DequantIQ.IQ2XS.Dequantize; return;
                case GgmlType.IQ2XXS:
                    blockSize = DequantIQ.IQ2XXS.BlockSize; bytesPerBlock = DequantIQ.IQ2XXS.BytesPerBlock;
                    dequant = DequantIQ.IQ2XXS.Dequantize; return;
                case GgmlType.IQ2S:
                    blockSize = DequantIQ.IQ2S.BlockSize; bytesPerBlock = DequantIQ.IQ2S.BytesPerBlock;
                    dequant = DequantIQ.IQ2S.Dequantize; return;
                case GgmlType.IQ1S:
                    blockSize = DequantIQ.IQ1S.BlockSize; bytesPerBlock = DequantIQ.IQ1S.BytesPerBlock;
                    dequant = DequantIQ.IQ1S.Dequantize; return;
                case GgmlType.IQ1M:
                    blockSize = DequantIQ.IQ1M.BlockSize; bytesPerBlock = DequantIQ.IQ1M.BytesPerBlock;
                    dequant = DequantIQ.IQ1M.Dequantize; return;
                default:
                    throw new NotSupportedException("Unsupported embedding type: " + type);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void MatVecMulByType(byte* weights, GgmlType type,
            ReadOnlySpan<float> input, Span<float> output, int outDim, int inDim)
        {
            switch (type)
            {
                case GgmlType.Q4_0:
                    TensorMath.MatVecMulQ4_0(weights, input, output, outDim, inDim);
                    return;
                case GgmlType.Q6K:
                    TensorMath.MatVecMulQ6K(weights, input, output, outDim, inDim);
                    return;
                case GgmlType.F32:
                    TensorMath.MatVecMul(new ReadOnlySpan<float>(weights, outDim * inDim), input, output, outDim, inDim);
                    return;
                case GgmlType.F16:
                    MatVecMulF16(weights, input, output, outDim, inDim);
                    return;

                // K-quant types with fused dot product
                case GgmlType.Q4_1:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ4_1.BlockSize, DequantQ4_1.BytesPerBlock, &DequantQ4_1.DotProduct);
                    return;
                case GgmlType.Q5_0:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ5_0.BlockSize, DequantQ5_0.BytesPerBlock, &DequantQ5_0.DotProduct);
                    return;
                case GgmlType.Q5_1:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ5_1.BlockSize, DequantQ5_1.BytesPerBlock, &DequantQ5_1.DotProduct);
                    return;
                case GgmlType.Q8_0:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ8_0.BlockSize, DequantQ8_0.BytesPerBlock, &DequantQ8_0.DotProduct);
                    return;
                case GgmlType.Q2K:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ2K.BlockSize, DequantQ2K.BytesPerBlock, &DequantQ2K.DotProduct);
                    return;
                case GgmlType.Q3K:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ3K.BlockSize, DequantQ3K.BytesPerBlock, &DequantQ3K.DotProduct);
                    return;
                case GgmlType.Q4K:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ4K.BlockSize, DequantQ4K.BytesPerBlock, &DequantQ4K.DotProduct);
                    return;
                case GgmlType.Q5K:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ5K.BlockSize, DequantQ5K.BytesPerBlock, &DequantQ5K.DotProduct);
                    return;
                case GgmlType.Q8K:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantQ8K.BlockSize, DequantQ8K.BytesPerBlock, &DequantQ8K.DotProduct);
                    return;

                // IQ family types
                case GgmlType.IQ4NL:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ4NL.BlockSize, DequantIQ.IQ4NL.BytesPerBlock, &DequantIQ.IQ4NL.DotProduct);
                    return;
                case GgmlType.IQ4XS:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ4XS.BlockSize, DequantIQ.IQ4XS.BytesPerBlock, &DequantIQ.IQ4XS.DotProduct);
                    return;
                case GgmlType.IQ3S:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ3S.BlockSize, DequantIQ.IQ3S.BytesPerBlock, &DequantIQ.IQ3S.DotProduct);
                    return;
                case GgmlType.IQ3XXS:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ3XXS.BlockSize, DequantIQ.IQ3XXS.BytesPerBlock, &DequantIQ.IQ3XXS.DotProduct);
                    return;
                case GgmlType.IQ2XS:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ2XS.BlockSize, DequantIQ.IQ2XS.BytesPerBlock, &DequantIQ.IQ2XS.DotProduct);
                    return;
                case GgmlType.IQ2XXS:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ2XXS.BlockSize, DequantIQ.IQ2XXS.BytesPerBlock, &DequantIQ.IQ2XXS.DotProduct);
                    return;
                case GgmlType.IQ2S:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ2S.BlockSize, DequantIQ.IQ2S.BytesPerBlock, &DequantIQ.IQ2S.DotProduct);
                    return;
                case GgmlType.IQ1S:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ1S.BlockSize, DequantIQ.IQ1S.BytesPerBlock, &DequantIQ.IQ1S.DotProduct);
                    return;
                case GgmlType.IQ1M:
                    TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim,
                        DequantIQ.IQ1M.BlockSize, DequantIQ.IQ1M.BytesPerBlock, &DequantIQ.IQ1M.DotProduct);
                    return;

                default:
                    throw new NotSupportedException("Unsupported weight quantization type: " + type);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MatVecMulF16(byte* weights, ReadOnlySpan<float> input,
            Span<float> output, int outDim, int inDim)
        {
            float[] dequantBuf = ArrayPool<float>.Shared.Rent(inDim);
            try
            {
                fixed (float* pInput = input)
                {
                    for (int row = 0; row < outDim; row++)
                    {
                        byte* rowPtr = weights + (long)row * inDim * 2;
                        for (int i = 0; i < inDim; i++)
                        {
                            dequantBuf[i] = DequantQ4_0.HalfToFloat(rowPtr[i * 2], rowPtr[i * 2 + 1]);
                        }
                        float dot = 0f;
                        for (int i = 0; i < inDim; i++)
                        {
                            dot += dequantBuf[i] * pInput[i];
                        }
                        output[row] = dot;
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(dequantBuf);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ReadOnlySpan<float> GetF32Weights(ReadOnlySpan<byte> data, int count)
        {
            return MemoryMarshal.Cast<byte, float>(data).Slice(0, count);
        }

        public void Dispose()
        {
            _weights.Dispose();
        }
    }
}
