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

namespace ChatNet.Core.Models.Phi
{
    /// <summary>
    /// Phi-3 model implementation. Uses SiLU-gated FFN, RMSNorm, and RoPE like Llama.
    /// Supports fused gate_up projection where gate and up tensors are stored as one.
    /// </summary>
    public sealed class PhiModel : IModel
    {
        private readonly PhiConfig _cfg;
        private readonly PhiWeights _weights;
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

        // Effective FFN hidden dimension, derived from actual tensor shapes.
        // This overrides _cfg.HiddenDim when GGUF metadata is wrong or missing.
        private readonly int _effectiveHiddenDim;

        public PhiModel(ModelConfig modelConfig, PhiWeights weights)
        {
            _modelConfig = modelConfig;
            _cfg = new PhiConfig(modelConfig);
            _weights = weights;

            // Use tensor-derived hiddenDim when it differs from config.
            // The ffn_down tensor's ne[0] is the ground truth for intermediate_size.
            _effectiveHiddenDim = weights.ActualFfnHiddenDim > 0
                ? weights.ActualFfnHiddenDim
                : _cfg.HiddenDim;

            if (_effectiveHiddenDim != _cfg.HiddenDim)
            {
                Console.Error.WriteLine("[WARN] PhiModel: overriding hiddenDim from " + _cfg.HiddenDim +
                    " to " + _effectiveHiddenDim + " based on ffn_down tensor dimensions." +
                    " GGUF metadata feed_forward_length is likely wrong or missing.");
            }

            if (DebugEnabled)
            {
                Console.Error.WriteLine("[DEBUG] PhiModel: arch=" + _modelConfig.Architecture +
                    " dim=" + _cfg.Dim + " hiddenDim=" + _effectiveHiddenDim +
                    " (config=" + _cfg.HiddenDim + ")" +
                    " layers=" + _cfg.LayerCount + " vocab=" + _cfg.VocabSize);
                Console.Error.WriteLine("[DEBUG] PhiModel: fusedQKV=" + weights.HasFusedQkv +
                    " fusedGateUp=" + weights.HasFusedGateUp +
                    " rotaryDim=" + _cfg.RotaryDim + " headDim=" + _cfg.HeadDim +
                    " kvMul=" + _cfg.KvMul + " ropeBase=" + _cfg.RopeFreqBase +
                    " rmsEps=" + _cfg.RmsNormEps);
                Console.Error.WriteLine("[DEBUG] PhiModel: embType=" + weights.EmbeddingType +
                    " outType=" + weights.OutputType);
                bool hasShort = _cfg.RopeScalingShortFactor != null;
                bool hasLong = _cfg.RopeScalingLongFactor != null;
                Console.Error.WriteLine("[DEBUG] PhiModel: ropeStyle=neox" +
                    " SuScaledRoPE=" + (hasShort || hasLong) +
                    " shortFactors=" + (hasShort ? _cfg.RopeScalingShortFactor!.Length + " elems" : "none") +
                    " longFactors=" + (hasLong ? _cfg.RopeScalingLongFactor!.Length + " elems" : "none") +
                    " origCtxLen=" + _cfg.OriginalContextLength);
                if (hasShort)
                {
                    var sf = _cfg.RopeScalingShortFactor!;
                    Console.Error.WriteLine("[DEBUG] PhiModel: shortFactor[0..3]=[" +
                        sf[0].ToString("F4") + "," + sf[Math.Min(1, sf.Length - 1)].ToString("F4") + "," +
                        sf[Math.Min(2, sf.Length - 1)].ToString("F4") + "," +
                        sf[Math.Min(3, sf.Length - 1)].ToString("F4") + "] last=" +
                        sf[sf.Length - 1].ToString("F4"));
                }

                // Warn if architecture is Phi-2 (different arch: LayerNorm, parallel residual, GELU, no gate)
                string archLower = _modelConfig.Architecture.ToLowerInvariant();
                if (archLower == "phi2" || archLower == "phi")
                {
                    Console.Error.WriteLine("[WARN] PhiModel: architecture '" + _modelConfig.Architecture +
                        "' detected. This forward path is Phi-3 (RMSNorm, SiLU-gated FFN, sequential residual). " +
                        "Phi-2 needs LayerNorm, GELU, parallel residual, non-gated FFN, and biases. " +
                        "If this is a Phi-2 model, output WILL be incorrect.");
                }
                if (weights.HasFusedQkv)
                    Console.Error.WriteLine("[DEBUG] PhiModel: attnQkv[0]=" + weights.AttnQkvType[0]);
                else
                    Console.Error.WriteLine("[DEBUG] PhiModel: attnQ[0]=" + weights.AttnQType[0] +
                        " attnK[0]=" + weights.AttnKType[0] + " attnV[0]=" + weights.AttnVType[0]);
                Console.Error.WriteLine("[DEBUG] PhiModel: attnOut[0]=" + weights.AttnOutputType[0] +
                    " ffnDown[0]=" + weights.FfnDownType[0]);
                if (weights.HasFusedGateUp)
                    Console.Error.WriteLine("[DEBUG] PhiModel: ffnGateUp[0]=" + weights.FfnGateUpType[0]);
                else
                    Console.Error.WriteLine("[DEBUG] PhiModel: ffnGate[0]=" + weights.FfnGateType[0] +
                        " ffnUp[0]=" + weights.FfnUpType[0]);
            }

            int maxSeq = _cfg.ContextLength;
            int kvDim = _cfg.KvDim;
            int dim = _cfg.Dim;
            int hiddenDim = _effectiveHiddenDim;

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
            int hiddenDim = _effectiveHiddenDim;
            int vocabSize = _cfg.VocabSize;

            for (int t = 0; t < tokenIds.Length; t++)
            {
                int tokenId = tokenIds[t];
                int pos = position + t;

                LoadEmbedding(tokenId, _x.AsSpan(0, dim));

                if (DebugEnabled && pos == 0)
                {
                    float embSum = 0f;
                    for (int ei = 0; ei < dim; ei++) embSum += _x[ei] * _x[ei];
                    Console.Error.WriteLine("[DEBUG] PhiEmb[token=" + tokenId + "] L2=" +
                        MathF.Sqrt(embSum).ToString("F6") +
                        " first5=[" + _x[0].ToString("F4") + "," + _x[1].ToString("F4") +
                        "," + _x[2].ToString("F4") + "," + _x[3].ToString("F4") +
                        "," + _x[4].ToString("F4") + "]");
                }

                for (int l = 0; l < layers; l++)
                {
                    ReadOnlySpan<float> attnNormW = GetF32Weights(_weights.GetAttnNormWeight(l), dim);
                    TensorMath.RmsNorm(_x.AsSpan(0, dim), attnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    if (DebugEnabled && pos == 0 && l == 0)
                    {
                        float normWL2 = 0f;
                        for (int ni = 0; ni < dim; ni++) normWL2 += attnNormW[ni] * attnNormW[ni];
                        float xNormL2 = 0f;
                        for (int ni = 0; ni < dim; ni++) xNormL2 += _xNorm[ni] * _xNorm[ni];
                        Console.Error.WriteLine("[DEBUG] PhiLayer 0 attnNorm: normW_L2=" +
                            MathF.Sqrt(normWL2).ToString("F4") +
                            " xNorm_L2=" + MathF.Sqrt(xNormL2).ToString("F4") +
                            " normW[0..4]=[" + attnNormW[0].ToString("F6") + "," +
                            attnNormW[1].ToString("F6") + "," + attnNormW[2].ToString("F6") + "," +
                            attnNormW[3].ToString("F6") + "," + attnNormW[4].ToString("F6") + "]");
                    }

                    unsafe
                    {
                        MatVecMulByType(_weights.GetAttnQWeight(l), _weights.AttnQType[l],
                            _xNorm.AsSpan(0, dim), _q.AsSpan(0, dim), dim, dim);
                        MatVecMulByType(_weights.GetAttnKWeight(l), _weights.AttnKType[l],
                            _xNorm.AsSpan(0, dim), _k.AsSpan(0, kvDim), kvDim, dim);
                        MatVecMulByType(_weights.GetAttnVWeight(l), _weights.AttnVType[l],
                            _xNorm.AsSpan(0, dim), _v.AsSpan(0, kvDim), kvDim, dim);
                    }

                    if (DebugEnabled && pos == 0 && l == 0)
                    {
                        // Print Q/K/V L2 norms after split, before RoPE
                        float qSum = 0f, kSum = 0f, vSum = 0f;
                        for (int qi = 0; qi < dim; qi++) qSum += _q[qi] * _q[qi];
                        for (int ki = 0; ki < kvDim; ki++) kSum += _k[ki] * _k[ki];
                        for (int vi = 0; vi < kvDim; vi++) vSum += _v[vi] * _v[vi];
                        Console.Error.WriteLine("[DEBUG] PhiQKV[layer0,pos0] preRoPE: Q_L2=" +
                            MathF.Sqrt(qSum).ToString("F4") + " K_L2=" +
                            MathF.Sqrt(kSum).ToString("F4") + " V_L2=" +
                            MathF.Sqrt(vSum).ToString("F4") +
                            " Q[0..3]=[" + _q[0].ToString("F4") + "," + _q[1].ToString("F4") +
                            "," + _q[2].ToString("F4") + "," + _q[3].ToString("F4") + "]");
                    }

                    // Phi-3 GGUF: llama.cpp does NOT permute Q/K weights for Phi-3.
                    // Weights are in original HuggingFace layout which uses neox-style
                    // (split-half) RoPE with partial rotation (rotaryDim = headDim/2 = 48).
                    TensorMath.ApplyRoPENeox(_q.AsSpan(0, dim), _k.AsSpan(0, kvDim),
                        pos, headDim, nHeads, nKvHeads, _cfg.RopeFreqBase, _cfg.RotaryDim);

                    if (DebugEnabled && pos == 0 && l == 0)
                    {
                        // Print Q/K L2 norms after RoPE to detect rotation issues
                        float qSum = 0f, kSum = 0f;
                        for (int qi = 0; qi < dim; qi++) qSum += _q[qi] * _q[qi];
                        for (int ki = 0; ki < kvDim; ki++) kSum += _k[ki] * _k[ki];
                        Console.Error.WriteLine("[DEBUG] PhiQKV[layer0,pos0] postRoPE: Q_L2=" +
                            MathF.Sqrt(qSum).ToString("F4") + " K_L2=" +
                            MathF.Sqrt(kSum).ToString("F4") +
                            " Q[0..3]=[" + _q[0].ToString("F4") + "," + _q[1].ToString("F4") +
                            "," + _q[2].ToString("F4") + "," + _q[3].ToString("F4") + "]");
                    }

                    int kvCacheLayerOffset = l * _cfg.ContextLength * kvDim;
                    int kvCachePos = kvCacheLayerOffset + pos * kvDim;
                    Array.Copy(_k, 0, _keyCache, kvCachePos, kvDim);
                    Array.Copy(_v, 0, _valueCache, kvCachePos, kvDim);

                    ComputeAttention(l, pos, headDim, nHeads, nKvHeads, kvMul, kvDim, dim);

                    if (DebugEnabled && pos == 0 && l <= 1)
                    {
                        float attnOutL2 = 0f;
                        for (int ai = 0; ai < dim; ai++) attnOutL2 += _attnOut[ai] * _attnOut[ai];
                        Console.Error.WriteLine("[DEBUG] PhiLayer " + l + " attnOut: L2=" +
                            MathF.Sqrt(attnOutL2).ToString("F4"));
                    }

                    unsafe
                    {
                        MatVecMulByType(_weights.GetAttnOutputWeight(l), _weights.AttnOutputType[l],
                            _attnOut.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim, dim);
                    }

                    if (DebugEnabled && pos == 0 && l <= 1)
                    {
                        float projL2 = 0f;
                        for (int xi = 0; xi < dim; xi++) projL2 += _ffnOut[xi] * _ffnOut[xi];
                        Console.Error.WriteLine("[DEBUG] PhiLayer " + l + " attnProj: L2=" +
                            MathF.Sqrt(projL2).ToString("F4"));
                    }

                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);

                    if (DebugEnabled && pos == 0 && l <= 1)
                    {
                        float xAfterAttn = 0f;
                        for (int xi = 0; xi < dim; xi++) xAfterAttn += _x[xi] * _x[xi];
                        Console.Error.WriteLine("[DEBUG] PhiLayer " + l + " postAttnResid: x_L2=" +
                            MathF.Sqrt(xAfterAttn).ToString("F4"));
                    }

                    ReadOnlySpan<float> ffnNormW = GetF32Weights(_weights.GetFfnNormWeight(l), dim);
                    TensorMath.RmsNorm(_x.AsSpan(0, dim), ffnNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                    if (DebugEnabled && pos == 0 && l == 0)
                    {
                        float fnormWL2 = 0f;
                        for (int ni = 0; ni < dim; ni++) fnormWL2 += ffnNormW[ni] * ffnNormW[ni];
                        float xNormL2 = 0f;
                        for (int ni = 0; ni < dim; ni++) xNormL2 += _xNorm[ni] * _xNorm[ni];
                        Console.Error.WriteLine("[DEBUG] PhiLayer 0 ffnNorm: normW_L2=" +
                            MathF.Sqrt(fnormWL2).ToString("F4") +
                            " xNorm_L2=" + MathF.Sqrt(xNormL2).ToString("F4") +
                            " normW[0..4]=[" + ffnNormW[0].ToString("F6") + "," +
                            ffnNormW[1].ToString("F6") + "," + ffnNormW[2].ToString("F6") + "," +
                            ffnNormW[3].ToString("F6") + "," + ffnNormW[4].ToString("F6") + "]");
                    }

                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnGateWeight(l), _weights.FfnGateType[l],
                            _xNorm.AsSpan(0, dim), _gate.AsSpan(0, hiddenDim), hiddenDim, dim);
                        MatVecMulByType(_weights.GetFfnUpWeight(l), _weights.FfnUpType[l],
                            _xNorm.AsSpan(0, dim), _up.AsSpan(0, hiddenDim), hiddenDim, dim);
                    }

                    if (DebugEnabled && pos == 0 && l == 0)
                    {
                        float gateSum = 0f, upSum = 0f;
                        for (int gi = 0; gi < hiddenDim; gi++) gateSum += _gate[gi] * _gate[gi];
                        for (int ui = 0; ui < hiddenDim; ui++) upSum += _up[ui] * _up[ui];
                        Console.Error.WriteLine("[DEBUG] PhiFFN[layer0,pos0] preSiLU: gate_L2=" +
                            MathF.Sqrt(gateSum).ToString("F4") + " up_L2=" +
                            MathF.Sqrt(upSum).ToString("F4") +
                            " gate[0..3]=[" + _gate[0].ToString("F4") + "," + _gate[1].ToString("F4") +
                            "," + _gate[2].ToString("F4") + "," + _gate[3].ToString("F4") + "]");
                    }

                    TensorMath.SiluElementwiseMul(_gate.AsSpan(0, hiddenDim), _up.AsSpan(0, hiddenDim), hiddenDim);

                    if (DebugEnabled && pos == 0 && l <= 1)
                    {
                        float siluL2 = 0f;
                        for (int gi = 0; gi < hiddenDim; gi++) siluL2 += _gate[gi] * _gate[gi];
                        Console.Error.WriteLine("[DEBUG] PhiLayer " + l + " postSiLU: L2=" +
                            MathF.Sqrt(siluL2).ToString("F4"));
                    }

                    unsafe
                    {
                        MatVecMulByType(_weights.GetFfnDownWeight(l), _weights.FfnDownType[l],
                            _gate.AsSpan(0, hiddenDim), _ffnOut.AsSpan(0, dim), dim, hiddenDim);
                    }

                    if (DebugEnabled && pos == 0 && l <= 1)
                    {
                        float ffnProjL2 = 0f;
                        for (int xi = 0; xi < dim; xi++) ffnProjL2 += _ffnOut[xi] * _ffnOut[xi];
                        Console.Error.WriteLine("[DEBUG] PhiLayer " + l + " ffnDownProj: L2=" +
                            MathF.Sqrt(ffnProjL2).ToString("F4"));
                    }

                    TensorMath.Add(_x.AsSpan(0, dim), _ffnOut.AsSpan(0, dim), dim);

                    // Log x L2 after every layer (not just first and last)
                    if (DebugEnabled && pos == 0)
                    {
                        float xSum = 0f;
                        for (int xi = 0; xi < dim; xi++) xSum += _x[xi] * _x[xi];
                        Console.Error.WriteLine("[DEBUG] PhiLayer " + l + ": x L2=" +
                            MathF.Sqrt(xSum).ToString("F6"));
                    }
                }

                ReadOnlySpan<float> finalNormW = GetF32Weights(_weights.GetFinalNormWeight(), dim);
                TensorMath.RmsNorm(_x.AsSpan(0, dim), finalNormW, _xNorm.AsSpan(0, dim), _cfg.RmsNormEps);

                if (t == tokenIds.Length - 1)
                {
                    unsafe
                    {
                        MatVecMulByType(_weights.GetOutputWeight(), _weights.OutputType,
                            _xNorm.AsSpan(0, dim), logits, vocabSize, dim);
                    }

                    if (DebugEnabled && pos <= 1)
                    {
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
                        Console.Error.WriteLine("[DEBUG] PhiLogits[pos=" + pos + "]: min=" +
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
            float scale = 1.0f / MathF.Sqrt(headDim);

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
                        pAttnScores[p] = score * scale;
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
