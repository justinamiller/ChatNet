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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ComputeAttention(int layer, int pos, int headDim, int nHeads, int nKvHeads, int kvMul, int kvDim, int dim)
        {
            int maxSeq = _cfg.ContextLength;
            int kvCacheLayerOffset = layer * maxSeq * kvDim;

            // Clear attention output
            _attnOut.AsSpan(0, dim).Clear();

            for (int h = 0; h < nHeads; h++)
            {
                int qOffset = h * headDim;
                int kvHead = h / kvMul;
                int kvOffset = kvHead * headDim;

                float scale = 1.0f / MathF.Sqrt(headDim);

                for (int p = 0; p <= pos; p++)
                {
                    int kCacheIdx = kvCacheLayerOffset + p * kvDim + kvOffset;
                    float score = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        score += _q[qOffset + d] * _keyCache[kCacheIdx + d];
                    }
                    _attnScores[p] = score * scale;
                }

                TensorMath.Softmax(_attnScores.AsSpan(), pos + 1);

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
            else if (_weights.EmbeddingType == GgmlType.Q4_0)
            {
                int blocksPerRow = dim / DequantQ4_0.BlockSize;
                int bytesPerRow = blocksPerRow * DequantQ4_0.BytesPerBlock;
                int rowOffset = tokenId * bytesPerRow;
                DequantQ4_0.Dequantize(embData.Slice(rowOffset, bytesPerRow), output, dim);
            }
            else if (_weights.EmbeddingType == GgmlType.Q6K)
            {
                int blocksPerRow = dim / DequantQ6K.BlockSize;
                int bytesPerRow = blocksPerRow * DequantQ6K.BytesPerBlock;
                int rowOffset = tokenId * bytesPerRow;
                DequantQ6K.Dequantize(embData.Slice(rowOffset, bytesPerRow), output, dim);
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
                throw new NotSupportedException("Unsupported embedding type: " + _weights.EmbeddingType);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void MatVecMulByType(byte* weights, GgmlType type,
            ReadOnlySpan<float> input, Span<float> output, int outDim, int inDim)
        {
            if (type == GgmlType.Q4_0)
            {
                TensorMath.MatVecMulQ4_0(weights, input, output, outDim, inDim);
            }
            else if (type == GgmlType.Q6K)
            {
                TensorMath.MatVecMulQ6K(weights, input, output, outDim, inDim);
            }
            else if (type == GgmlType.F32)
            {
                ReadOnlySpan<float> fWeights = new ReadOnlySpan<float>(weights, outDim * inDim);
                TensorMath.MatVecMul(fWeights, input, output, outDim, inDim);
            }
            else if (type == GgmlType.F16)
            {
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
            else
            {
                throw new NotSupportedException("Unsupported weight quantization type: " + type);
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
