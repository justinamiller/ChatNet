using System;
using System.Buffers;
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

        public ModelConfig Config => _modelConfig;

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
        }

        /// <summary>
        /// Run forward pass for a sequence of tokens starting at the given position.
        /// For autoregressive generation, tokenIds.Length is typically 1, with position incrementing.
        /// Logits output is for the LAST token in the sequence.
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

                // Step 1: Token embedding - copy embedding vector to x
                LoadEmbedding(tokenId, _x.AsSpan(0, dim));

                // Step 2: Process each transformer layer
                for (int l = 0; l < layers; l++)
                {
                    // 2a: RMS Norm before attention
                    ReadOnlySpan<float> attnNormW = GetF32Weights(_weights.GetAttnNormWeight(l), dim);
                    TensorMath.RmsNorm(_x.AsSpan(0, dim), attnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    // 2b: QKV projections
                    unsafe
                    {
                        // Q = xNorm @ Wq  (Wq shape: [dim, dim])
                        MatVecMulByType(_weights.GetAttnQWeight(l), _weights.AttnQType[l],
                            _xNorm.AsSpan(0, dim), _q.AsSpan(0, dim), dim, dim);

                        // K = xNorm @ Wk  (Wk shape: [kvDim, dim])
                        MatVecMulByType(_weights.GetAttnKWeight(l), _weights.AttnKType[l],
                            _xNorm.AsSpan(0, dim), _k.AsSpan(0, kvDim), kvDim, dim);

                        // V = xNorm @ Wv  (Wv shape: [kvDim, dim])
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

                    // 2g: Residual connection: x = x + attnResult
                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);

                    // 2h: RMS Norm before FFN
                    ReadOnlySpan<float> ffnNormW = GetF32Weights(_weights.GetFfnNormWeight(l), dim);
                    TensorMath.RmsNorm(_x.AsSpan(0, dim), ffnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    // 2i: FFN (SiLU-gated)
                    unsafe
                    {
                        // gate = xNorm @ W1 (gate projection) [hiddenDim, dim]
                        MatVecMulByType(_weights.GetFfnGateWeight(l), _weights.FfnGateType[l],
                            _xNorm.AsSpan(0, dim), _gate.AsSpan(0, hiddenDim), hiddenDim, dim);

                        // up = xNorm @ W3 (up projection) [hiddenDim, dim]
                        MatVecMulByType(_weights.GetFfnUpWeight(l), _weights.FfnUpType[l],
                            _xNorm.AsSpan(0, dim), _up.AsSpan(0, hiddenDim), hiddenDim, dim);
                    }

                    // hidden = silu(gate) * up
                    TensorMath.SiluElementwiseMul(_gate.AsSpan(0, hiddenDim), _up.AsSpan(0, hiddenDim), hiddenDim);

                    // ffnOut = hidden @ W2 (down projection) [dim, hiddenDim]
                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnDownWeight(l), _weights.FfnDownType[l],
                            _gate.AsSpan(0, hiddenDim), _ffnOut.AsSpan(0, dim), dim, hiddenDim);
                    }

                    // 2j: Residual connection: x = x + ffnOut
                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);
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
                }
            }
        }

        /// <summary>
        /// Compute multi-head attention with grouped-query attention (GQA).
        /// Results are written to _attnOut.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ComputeAttention(int layer, int pos, int headDim, int nHeads, int nKvHeads, int kvMul, int kvDim, int dim)
        {
            int maxSeq = _cfg.ContextLength;
            int kvCacheLayerOffset = layer * maxSeq * kvDim;

            // Clear attention output
            _attnOut.AsSpan(0, dim).Clear();

            // For each query head
            for (int h = 0; h < nHeads; h++)
            {
                int qOffset = h * headDim;
                int kvHead = h / kvMul; // GQA: which KV head this query head uses
                int kvOffset = kvHead * headDim;

                // Compute attention scores: Q @ K^T / sqrt(headDim) for all cached positions
                float scale = 1.0f / MathF.Sqrt(headDim);

                for (int p = 0; p <= pos; p++)
                {
                    // Score = dot(Q[h], K_cached[p, kvHead]) / sqrt(headDim)
                    int kCacheIdx = kvCacheLayerOffset + p * kvDim + kvOffset;
                    float score = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        score += _q[qOffset + d] * _keyCache[kCacheIdx + d];
                    }
                    _attnScores[p] = score * scale;
                }

                // Softmax over attention scores [0..pos]
                TensorMath.Softmax(_attnScores.AsSpan(), pos + 1);

                // Weighted sum of values: attnOut[h] += score[p] * V_cached[p, kvHead]
                for (int p = 0; p <= pos; p++)
                {
                    float attnWeight = _attnScores[p];
                    if (attnWeight == 0f) continue;
                    int vCacheIdx = kvCacheLayerOffset + p * kvDim + kvOffset;
                    for (int d = 0; d < headDim; d++)
                    {
                        _attnOut[qOffset + d] += attnWeight * _valueCache[vCacheIdx + d];
                    }
                }
            }
        }

        /// <summary>
        /// Load embedding vector for a token. Handles F32 and Q4_0 embeddings.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LoadEmbedding(int tokenId, Span<float> output)
        {
            int dim = _cfg.Dim;
            ReadOnlySpan<byte> embData = _weights.GetEmbeddingData();

            if (_weights.EmbeddingType == GgmlType.F32)
            {
                // F32 embedding: direct copy
                ReadOnlySpan<float> allEmb = MemoryMarshal.Cast<byte, float>(embData);
                int offset = tokenId * dim;
                allEmb.Slice(offset, dim).CopyTo(output);
            }
            else if (_weights.EmbeddingType == GgmlType.Q4_0)
            {
                // Q4_0 embedding: dequantize the row for this token
                int blocksPerRow = dim / DequantQ4_0.BlockSize;
                int bytesPerRow = blocksPerRow * DequantQ4_0.BytesPerBlock;
                int rowOffset = tokenId * bytesPerRow;
                DequantQ4_0.Dequantize(embData.Slice(rowOffset, bytesPerRow), output, dim);
            }
            else if (_weights.EmbeddingType == GgmlType.F16)
            {
                // F16 embedding: convert half to float
                int offset = tokenId * dim * 2;
                for (int i = 0; i < dim; i++)
                {
                    output[i] = DequantQ4_0.HalfToFloat(embData[offset + i * 2], embData[offset + i * 2 + 1]);
                }
            }
        }

        /// <summary>
        /// Dispatch matrix-vector multiply based on tensor quantization type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void MatVecMulByType(byte* weights, GgmlType type,
            ReadOnlySpan<float> input, Span<float> output, int outDim, int inDim)
        {
            if (type == GgmlType.Q4_0)
            {
                TensorMath.MatVecMulQ4_0(weights, input, output, outDim, inDim);
            }
            else if (type == GgmlType.F32)
            {
                ReadOnlySpan<float> fWeights = new ReadOnlySpan<float>(weights, outDim * inDim);
                TensorMath.MatVecMul(fWeights, input, output, outDim, inDim);
            }
            else if (type == GgmlType.F16)
            {
                // Dequantize F16 on the fly - row by row
                float[] dequantBuf = ArrayPool<float>.Shared.Rent(inDim);
                try
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
                            dot += dequantBuf[i] * input[i];
                        }
                        output[row] = dot;
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(dequantBuf);
                }
            }
        }

        /// <summary>
        /// Reinterpret raw byte data as float span (for F32 norm weights).
        /// </summary>
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
