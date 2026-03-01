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

namespace ChatNet.Core.Models.Gemma
{
    /// <summary>
    /// Gemma model implementation. Key differences from Llama:
    /// 1) Embedding scaling by sqrt(dim)
    /// 2) RMSNorm weights have +1 offset (stored as delta from 1.0)
    /// 3) GELU activation instead of SiLU in the gated FFN
    /// 4) Output may be tied to embeddings (no separate output.weight)
    /// Gemma 2 additions:
    /// 5) Post-attention and post-FFN norms
    /// 6) Attention logit soft-capping
    /// 7) Final logit soft-capping
    /// </summary>
    public sealed class GemmaModel : IModel
    {
        private readonly GemmaConfig _cfg;
        private readonly GemmaWeights _weights;
        private readonly ModelConfig _modelConfig;

        private readonly float[] _keyCache;
        private readonly float[] _valueCache;
        private readonly float[] _x;
        private readonly float[] _xNorm;
        private readonly float[] _q;
        private readonly float[] _k;
        private readonly float[] _v;
        private readonly float[] _attnOut;
        private readonly float[] _gate;
        private readonly float[] _up;
        private readonly float[] _ffnOut;
        private readonly float[] _attnScores;

        public ModelConfig Config => _modelConfig;
        public static bool DebugEnabled { get; set; }

        public GemmaModel(ModelConfig modelConfig, GemmaWeights weights)
        {
            _modelConfig = modelConfig;
            _cfg = new GemmaConfig(modelConfig);
            _weights = weights;

            int maxSeq = _cfg.ContextLength;
            int kvDim = _cfg.KvDim;
            int dim = _cfg.Dim;
            int hiddenDim = _cfg.HiddenDim;

            _keyCache = new float[_cfg.LayerCount * maxSeq * kvDim];
            _valueCache = new float[_cfg.LayerCount * maxSeq * kvDim];

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
            float embScale = _cfg.EmbeddingScale;
            bool hasPostNorms = _cfg.HasPostNorms;
            float attnSoftcap = _cfg.AttnLogitSoftcap;
            float finalSoftcap = _cfg.FinalLogitSoftcap;

            for (int t = 0; t < tokenIds.Length; t++)
            {
                int tokenId = tokenIds[t];
                int pos = position + t;

                // Gemma: embedding scaled by sqrt(dim)
                LoadEmbedding(tokenId, _x.AsSpan(0, dim));
                ScaleVector(_x.AsSpan(0, dim), embScale);

                for (int l = 0; l < layers; l++)
                {
                    // Pre-attention RMSNorm with +1 offset
                    ReadOnlySpan<float> attnNormW = GetF32Weights(_weights.GetAttnNormWeight(l), dim);
                    TensorMath.RmsNormWithOffset(_x.AsSpan(0, dim), attnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    unsafe
                    {
                        MatVecMulByType(_weights.GetAttnQWeight(l), _weights.AttnQType[l],
                            _xNorm.AsSpan(0, dim), _q.AsSpan(0, dim), dim, dim);
                        MatVecMulByType(_weights.GetAttnKWeight(l), _weights.AttnKType[l],
                            _xNorm.AsSpan(0, dim), _k.AsSpan(0, kvDim), kvDim, dim);
                        MatVecMulByType(_weights.GetAttnVWeight(l), _weights.AttnVType[l],
                            _xNorm.AsSpan(0, dim), _v.AsSpan(0, kvDim), kvDim, dim);
                    }

                    TensorMath.ApplyRoPE(_q.AsSpan(0, dim), _k.AsSpan(0, kvDim),
                        pos, headDim, nHeads, nKvHeads, _cfg.RopeFreqBase);

                    int kvCacheLayerOffset = l * _cfg.ContextLength * kvDim;
                    int kvCachePos = kvCacheLayerOffset + pos * kvDim;
                    Array.Copy(_k, 0, _keyCache, kvCachePos, kvDim);
                    Array.Copy(_v, 0, _valueCache, kvCachePos, kvDim);

                    ComputeAttention(l, pos, headDim, nHeads, nKvHeads, kvMul, kvDim, dim, attnSoftcap);

                    unsafe
                    {
                        MatVecMulByType(_weights.GetAttnOutputWeight(l), _weights.AttnOutputType[l],
                            _attnOut.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim, dim);
                    }

                    // Gemma 2: post-attention norm
                    if (hasPostNorms)
                    {
                        ReadOnlySpan<float> postAttnW = GetF32Weights(_weights.GetPostAttnNormWeight(l), dim);
                        TensorMath.RmsNormWithOffset(_ffnOut.AsSpan(0, dim), postAttnW, _ffnOut.AsSpan(0, dim), _cfg.RmsNormEps);
                    }

                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);

                    // Pre-FFN RMSNorm with +1 offset
                    ReadOnlySpan<float> ffnNormW = GetF32Weights(_weights.GetFfnNormWeight(l), dim);
                    TensorMath.RmsNormWithOffset(_x.AsSpan(0, dim), ffnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnGateWeight(l), _weights.FfnGateType[l],
                            _xNorm.AsSpan(0, dim), _gate.AsSpan(0, hiddenDim), hiddenDim, dim);
                        MatVecMulByType(_weights.GetFfnUpWeight(l), _weights.FfnUpType[l],
                            _xNorm.AsSpan(0, dim), _up.AsSpan(0, hiddenDim), hiddenDim, dim);
                    }

                    // GELU activation instead of SiLU
                    TensorMath.GeluElementwiseMul(_gate.AsSpan(0, hiddenDim), _up.AsSpan(0, hiddenDim), hiddenDim);

                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnDownWeight(l), _weights.FfnDownType[l],
                            _gate.AsSpan(0, hiddenDim), _ffnOut.AsSpan(0, dim), dim, hiddenDim);
                    }

                    // Gemma 2: post-FFN norm
                    if (hasPostNorms)
                    {
                        ReadOnlySpan<float> postFfnW = GetF32Weights(_weights.GetPostFfnNormWeight(l), dim);
                        TensorMath.RmsNormWithOffset(_ffnOut.AsSpan(0, dim), postFfnW, _ffnOut.AsSpan(0, dim), _cfg.RmsNormEps);
                    }

                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);
                }

                // Final norm with +1 offset
                ReadOnlySpan<float> finalNormW = GetF32Weights(_weights.GetFinalNormWeight(), dim);
                TensorMath.RmsNormWithOffset(_x.AsSpan(0, dim), finalNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                if (t == tokenIds.Length - 1)
                {
                    unsafe
                    {
                        MatVecMulByType(_weights.GetOutputWeight(), _weights.OutputType,
                            _xNorm.AsSpan(0, dim), logits, vocabSize, dim);
                    }

                    // Gemma 2: final logit soft-capping
                    if (finalSoftcap > 0f)
                        ApplySoftcap(logits.Slice(0, vocabSize), finalSoftcap);
                }
            }
        }

        /// <summary>
        /// Logit soft-capping: logits = cap * tanh(logits / cap)
        /// Constrains logits to [-cap, cap] range.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ApplySoftcap(Span<float> logits, float cap)
        {
            float invCap = 1.0f / cap;
            for (int i = 0; i < logits.Length; i++)
                logits[i] = cap * MathF.Tanh(logits[i] * invCap);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ScaleVector(Span<float> vec, float scale)
        {
            int n = vec.Length;
            fixed (float* p = vec)
            {
                int i = 0;
                if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
                {
                    var vScale = new Vector<float>(scale);
                    int vecLen = Vector<float>.Count;
                    int limit = n - vecLen + 1;
                    for (; i < limit; i += vecLen)
                        *(Vector<float>*)(p + i) = *(Vector<float>*)(p + i) * vScale;
                }
                for (; i < n; i++)
                    p[i] *= scale;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private unsafe void ComputeAttention(int layer, int pos, int headDim, int nHeads, int nKvHeads, int kvMul, int kvDim, int dim, float softcap)
        {
            int maxSeq = _cfg.ContextLength;
            int kvCacheLayerOffset = layer * maxSeq * kvDim;
            int seqLen = pos + 1;
            float scale = 1.0f / MathF.Sqrt(headDim);
            bool useSoftcap = softcap > 0f;
            float invSoftcap = useSoftcap ? 1.0f / softcap : 0f;

            fixed (float* pQ = _q)
            fixed (float* pKeyCache = _keyCache)
            fixed (float* pValueCache = _valueCache)
            fixed (float* pAttnScores = _attnScores)
            fixed (float* pAttnOut = _attnOut)
            {
                new Span<float>(pAttnOut, dim).Clear();

                for (int h = 0; h < nHeads; h++)
                {
                    int qOffset = h * headDim;
                    int kvHead = h / kvMul;
                    int kvOffset = kvHead * headDim;
                    float* qPtr = pQ + qOffset;

                    for (int p = 0; p < seqLen; p++)
                    {
                        float* kPtr = pKeyCache + kvCacheLayerOffset + p * kvDim + kvOffset;
                        float score = 0f;
                        int d = 0;
                        if (Vector.IsHardwareAccelerated && headDim >= Vector<float>.Count)
                        {
                            int vecLen = Vector<float>.Count;
                            var vAcc0 = Vector<float>.Zero;
                            var vAcc1 = Vector<float>.Zero;
                            int limit = headDim - 2 * vecLen + 1;
                            for (; d < limit; d += 2 * vecLen)
                            {
                                vAcc0 += *(Vector<float>*)(qPtr + d) * *(Vector<float>*)(kPtr + d);
                                vAcc1 += *(Vector<float>*)(qPtr + d + vecLen) * *(Vector<float>*)(kPtr + d + vecLen);
                            }
                            int singleLimit = headDim - vecLen + 1;
                            for (; d < singleLimit; d += vecLen)
                                vAcc0 += *(Vector<float>*)(qPtr + d) * *(Vector<float>*)(kPtr + d);
                            score = VectorSumFast(vAcc0 + vAcc1);
                        }
                        for (; d < headDim; d++) score += qPtr[d] * kPtr[d];
                        score *= scale;

                        // Gemma 2: attention logit soft-capping
                        if (useSoftcap)
                            score = softcap * MathF.Tanh(score * invSoftcap);

                        pAttnScores[p] = score;
                    }

                    TensorMath.Softmax(new Span<float>(pAttnScores, seqLen), seqLen);

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
                                *(Vector<float>*)(outPtr + d) += *(Vector<float>*)(vPtr + d) * vWeight;
                        }
                        for (; d < headDim; d++) outPtr[d] += attnWeight * vPtr[d];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float VectorSumFast(Vector<float> v)
        {
            float sum = 0f;
            for (int i = 0; i < Vector<float>.Count; i++) sum += v[i];
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
                allEmb.Slice(tokenId * dim, dim).CopyTo(output);
            }
            else if (_weights.EmbeddingType == GgmlType.F16)
            {
                int offset = tokenId * dim * 2;
                for (int i = 0; i < dim; i++)
                    output[i] = DequantQ4_0.HalfToFloat(embData[offset + i * 2], embData[offset + i * 2 + 1]);
            }
            else
            {
                GetDequantInfo(_weights.EmbeddingType, out int blockSize, out int bytesPerBlock, out var dequant);
                int blocksPerRow = dim / blockSize;
                int bytesPerRow = blocksPerRow * bytesPerBlock;
                dequant(embData.Slice(tokenId * bytesPerRow, bytesPerRow), output, dim);
            }
        }

        private delegate void DequantizeDelegate(ReadOnlySpan<byte> data, Span<float> output, int count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void GetDequantInfo(GgmlType type, out int blockSize, out int bytesPerBlock, out DequantizeDelegate dequant)
        {
            switch (type)
            {
                case GgmlType.Q4_0: blockSize = DequantQ4_0.BlockSize; bytesPerBlock = DequantQ4_0.BytesPerBlock; dequant = DequantQ4_0.Dequantize; return;
                case GgmlType.Q4_1: blockSize = DequantQ4_1.BlockSize; bytesPerBlock = DequantQ4_1.BytesPerBlock; dequant = DequantQ4_1.Dequantize; return;
                case GgmlType.Q5_0: blockSize = DequantQ5_0.BlockSize; bytesPerBlock = DequantQ5_0.BytesPerBlock; dequant = DequantQ5_0.Dequantize; return;
                case GgmlType.Q5_1: blockSize = DequantQ5_1.BlockSize; bytesPerBlock = DequantQ5_1.BytesPerBlock; dequant = DequantQ5_1.Dequantize; return;
                case GgmlType.Q8_0: blockSize = DequantQ8_0.BlockSize; bytesPerBlock = DequantQ8_0.BytesPerBlock; dequant = DequantQ8_0.Dequantize; return;
                case GgmlType.Q2K: blockSize = DequantQ2K.BlockSize; bytesPerBlock = DequantQ2K.BytesPerBlock; dequant = DequantQ2K.Dequantize; return;
                case GgmlType.Q3K: blockSize = DequantQ3K.BlockSize; bytesPerBlock = DequantQ3K.BytesPerBlock; dequant = DequantQ3K.Dequantize; return;
                case GgmlType.Q4K: blockSize = DequantQ4K.BlockSize; bytesPerBlock = DequantQ4K.BytesPerBlock; dequant = DequantQ4K.Dequantize; return;
                case GgmlType.Q5K: blockSize = DequantQ5K.BlockSize; bytesPerBlock = DequantQ5K.BytesPerBlock; dequant = DequantQ5K.Dequantize; return;
                case GgmlType.Q6K: blockSize = DequantQ6K.BlockSize; bytesPerBlock = DequantQ6K.BytesPerBlock; dequant = DequantQ6K.Dequantize; return;
                case GgmlType.Q8K: blockSize = DequantQ8K.BlockSize; bytesPerBlock = DequantQ8K.BytesPerBlock; dequant = DequantQ8K.Dequantize; return;
                default: throw new NotSupportedException("Unsupported embedding type: " + type);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void MatVecMulByType(byte* weights, GgmlType type,
            ReadOnlySpan<float> input, Span<float> output, int outDim, int inDim)
        {
            switch (type)
            {
                case GgmlType.Q4_0: TensorMath.MatVecMulQ4_0(weights, input, output, outDim, inDim); return;
                case GgmlType.Q6K: TensorMath.MatVecMulQ6K(weights, input, output, outDim, inDim); return;
                case GgmlType.F32: TensorMath.MatVecMul(new ReadOnlySpan<float>(weights, outDim * inDim), input, output, outDim, inDim); return;
                case GgmlType.F16: MatVecMulF16(weights, input, output, outDim, inDim); return;
                case GgmlType.Q4_1: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ4_1.BlockSize, DequantQ4_1.BytesPerBlock, &DequantQ4_1.DotProduct); return;
                case GgmlType.Q5_0: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ5_0.BlockSize, DequantQ5_0.BytesPerBlock, &DequantQ5_0.DotProduct); return;
                case GgmlType.Q5_1: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ5_1.BlockSize, DequantQ5_1.BytesPerBlock, &DequantQ5_1.DotProduct); return;
                case GgmlType.Q8_0: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ8_0.BlockSize, DequantQ8_0.BytesPerBlock, &DequantQ8_0.DotProduct); return;
                case GgmlType.Q2K: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ2K.BlockSize, DequantQ2K.BytesPerBlock, &DequantQ2K.DotProduct); return;
                case GgmlType.Q3K: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ3K.BlockSize, DequantQ3K.BytesPerBlock, &DequantQ3K.DotProduct); return;
                case GgmlType.Q4K: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ4K.BlockSize, DequantQ4K.BytesPerBlock, &DequantQ4K.DotProduct); return;
                case GgmlType.Q5K: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ5K.BlockSize, DequantQ5K.BytesPerBlock, &DequantQ5K.DotProduct); return;
                case GgmlType.Q8K: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantQ8K.BlockSize, DequantQ8K.BytesPerBlock, &DequantQ8K.DotProduct); return;
                case GgmlType.IQ4NL: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ4NL.BlockSize, DequantIQ.IQ4NL.BytesPerBlock, &DequantIQ.IQ4NL.DotProduct); return;
                case GgmlType.IQ4XS: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ4XS.BlockSize, DequantIQ.IQ4XS.BytesPerBlock, &DequantIQ.IQ4XS.DotProduct); return;
                case GgmlType.IQ3S: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ3S.BlockSize, DequantIQ.IQ3S.BytesPerBlock, &DequantIQ.IQ3S.DotProduct); return;
                case GgmlType.IQ3XXS: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ3XXS.BlockSize, DequantIQ.IQ3XXS.BytesPerBlock, &DequantIQ.IQ3XXS.DotProduct); return;
                case GgmlType.IQ2XS: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ2XS.BlockSize, DequantIQ.IQ2XS.BytesPerBlock, &DequantIQ.IQ2XS.DotProduct); return;
                case GgmlType.IQ2XXS: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ2XXS.BlockSize, DequantIQ.IQ2XXS.BytesPerBlock, &DequantIQ.IQ2XXS.DotProduct); return;
                case GgmlType.IQ2S: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ2S.BlockSize, DequantIQ.IQ2S.BytesPerBlock, &DequantIQ.IQ2S.DotProduct); return;
                case GgmlType.IQ1S: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ1S.BlockSize, DequantIQ.IQ1S.BytesPerBlock, &DequantIQ.IQ1S.DotProduct); return;
                case GgmlType.IQ1M: TensorMath.MatVecMulGeneric(weights, input, output, outDim, inDim, DequantIQ.IQ1M.BlockSize, DequantIQ.IQ1M.BytesPerBlock, &DequantIQ.IQ1M.DotProduct); return;
                default: throw new NotSupportedException("Unsupported weight quantization type: " + type);
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
                            dequantBuf[i] = DequantQ4_0.HalfToFloat(rowPtr[i * 2], rowPtr[i * 2 + 1]);
                        float dot = 0f;
                        for (int i = 0; i < inDim; i++) dot += dequantBuf[i] * pInput[i];
                        output[row] = dot;
                    }
                }
            }
            finally { ArrayPool<float>.Shared.Return(dequantBuf); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ReadOnlySpan<float> GetF32Weights(ReadOnlySpan<byte> data, int count)
        {
            return MemoryMarshal.Cast<byte, float>(data).Slice(0, count);
        }

        public void Dispose() { _weights.Dispose(); }
    }
}
