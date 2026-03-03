using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;

namespace ChatNet.Core.Models.Qwen
{
    /// <summary>
    /// Weight tensor storage for Qwen2 model.
    /// Similar to LlamaWeights but supports optional QKV biases.
    /// All pointers are resolved at construction time for zero-alloc forward pass.
    /// </summary>
    public sealed unsafe class QwenWeights : IDisposable
    {
        // Embedding
        public GgmlType EmbeddingType { get; private set; }
        private byte* _embeddingPtr;
        private int _embeddingByteSize;

        // Per-layer cached pointers
        private readonly byte*[] _attnNormWeight;
        private readonly byte*[] _ffnNormWeight;
        private readonly byte*[] _attnQWeight;
        private readonly byte*[] _attnKWeight;
        private readonly byte*[] _attnVWeight;
        private readonly byte*[] _attnOutputWeight;
        private readonly byte*[] _ffnGateWeight;
        private readonly byte*[] _ffnUpWeight;
        private readonly byte*[] _ffnDownWeight;

        // Optional QKV biases (Qwen2 specific)
        private readonly float*[] _attnQBias;
        private readonly float*[] _attnKBias;
        private readonly float*[] _attnVBias;
        private readonly bool _hasBias;

        // Per-layer quantization types
        public GgmlType[] AttnQType { get; }
        public GgmlType[] AttnKType { get; }
        public GgmlType[] AttnVType { get; }
        public GgmlType[] AttnOutputType { get; }
        public GgmlType[] FfnGateType { get; }
        public GgmlType[] FfnUpType { get; }
        public GgmlType[] FfnDownType { get; }

        // Per-layer norm data sizes
        private readonly int[] _attnNormSize;
        private readonly int[] _ffnNormSize;

        // Output projection
        public GgmlType OutputType { get; private set; }
        private byte* _outputWeight;
        private byte* _finalNormWeight;
        private int _finalNormSize;

        public bool HasBias => _hasBias;

        public QwenWeights(MemoryMappedWeights weights, QwenConfig config)
        {
            int layers = config.LayerCount;

            _attnNormWeight = new byte*[layers];
            _ffnNormWeight = new byte*[layers];
            _attnQWeight = new byte*[layers];
            _attnKWeight = new byte*[layers];
            _attnVWeight = new byte*[layers];
            _attnOutputWeight = new byte*[layers];
            _ffnGateWeight = new byte*[layers];
            _ffnUpWeight = new byte*[layers];
            _ffnDownWeight = new byte*[layers];

            _attnQBias = new float*[layers];
            _attnKBias = new float*[layers];
            _attnVBias = new float*[layers];

            AttnQType = new GgmlType[layers];
            AttnKType = new GgmlType[layers];
            AttnVType = new GgmlType[layers];
            AttnOutputType = new GgmlType[layers];
            FfnGateType = new GgmlType[layers];
            FfnUpType = new GgmlType[layers];
            FfnDownType = new GgmlType[layers];

            _attnNormSize = new int[layers];
            _ffnNormSize = new int[layers];

            _hasBias = ResolveAll(weights, config);
        }

        private bool ResolveAll(MemoryMappedWeights w, QwenConfig config)
        {
            int layers = config.LayerCount;
            bool hasBias = false;

            // Embedding
            GgufTensorInfo embInfo = w.GetTensorInfo(QwenTensorNames.Embedding);
            EmbeddingType = embInfo.Type;
            _embeddingPtr = w.GetTensorPointer(QwenTensorNames.Embedding);
            _embeddingByteSize = (int)embInfo.ByteSize;

            // Output weight
            if (w.HasTensor(QwenTensorNames.Output))
            {
                OutputType = w.GetTensorInfo(QwenTensorNames.Output).Type;
                _outputWeight = w.GetTensorPointer(QwenTensorNames.Output);
            }
            else
            {
                OutputType = EmbeddingType;
                _outputWeight = _embeddingPtr;
            }

            // Final norm
            _finalNormWeight = w.GetTensorPointer(QwenTensorNames.OutputNorm);
            _finalNormSize = (int)w.GetTensorInfo(QwenTensorNames.OutputNorm).ByteSize;

            // Per-layer weights
            for (int l = 0; l < layers; l++)
            {
                string prefix = QwenTensorNames.BlockPrefix + l.ToString();

                string attnNormName = prefix + QwenTensorNames.AttnNormSuffix;
                _attnNormWeight[l] = w.GetTensorPointer(attnNormName);
                _attnNormSize[l] = (int)w.GetTensorInfo(attnNormName).ByteSize;

                string ffnNormName = prefix + QwenTensorNames.FfnNormSuffix;
                _ffnNormWeight[l] = w.GetTensorPointer(ffnNormName);
                _ffnNormSize[l] = (int)w.GetTensorInfo(ffnNormName).ByteSize;

                string aqName = prefix + QwenTensorNames.AttnQSuffix;
                _attnQWeight[l] = w.GetTensorPointer(aqName);
                AttnQType[l] = w.GetTensorInfo(aqName).Type;

                string akName = prefix + QwenTensorNames.AttnKSuffix;
                _attnKWeight[l] = w.GetTensorPointer(akName);
                AttnKType[l] = w.GetTensorInfo(akName).Type;

                string avName = prefix + QwenTensorNames.AttnVSuffix;
                _attnVWeight[l] = w.GetTensorPointer(avName);
                AttnVType[l] = w.GetTensorInfo(avName).Type;

                string aoName = prefix + QwenTensorNames.AttnOutputSuffix;
                _attnOutputWeight[l] = w.GetTensorPointer(aoName);
                AttnOutputType[l] = w.GetTensorInfo(aoName).Type;

                string fgName = prefix + QwenTensorNames.FfnGateSuffix;
                _ffnGateWeight[l] = w.GetTensorPointer(fgName);
                FfnGateType[l] = w.GetTensorInfo(fgName).Type;

                string fuName = prefix + QwenTensorNames.FfnUpSuffix;
                _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                FfnUpType[l] = w.GetTensorInfo(fuName).Type;

                string fdName = prefix + QwenTensorNames.FfnDownSuffix;
                _ffnDownWeight[l] = w.GetTensorPointer(fdName);
                FfnDownType[l] = w.GetTensorInfo(fdName).Type;

                // Optional biases
                string qBiasName = prefix + QwenTensorNames.AttnQBiasSuffix;
                string kBiasName = prefix + QwenTensorNames.AttnKBiasSuffix;
                string vBiasName = prefix + QwenTensorNames.AttnVBiasSuffix;

                if (w.HasTensor(qBiasName))
                {
                    _attnQBias[l] = (float*)w.GetTensorPointer(qBiasName);
                    _attnKBias[l] = (float*)w.GetTensorPointer(kBiasName);
                    _attnVBias[l] = (float*)w.GetTensorPointer(vBiasName);
                    hasBias = true;
                }
            }

            return hasBias;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetEmbeddingData() => new ReadOnlySpan<byte>(_embeddingPtr, _embeddingByteSize);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetAttnNormWeight(int layer) =>
            new ReadOnlySpan<byte>(_attnNormWeight[layer], _attnNormSize[layer]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetFfnNormWeight(int layer) =>
            new ReadOnlySpan<byte>(_ffnNormWeight[layer], _ffnNormSize[layer]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetFinalNormWeight() =>
            new ReadOnlySpan<byte>(_finalNormWeight, _finalNormSize);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnQWeight(int layer) => _attnQWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnKWeight(int layer) => _attnKWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnVWeight(int layer) => _attnVWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnOutputWeight(int layer) => _attnOutputWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnGateWeight(int layer) => _ffnGateWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnUpWeight(int layer) => _ffnUpWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnDownWeight(int layer) => _ffnDownWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetOutputWeight() => _outputWeight;

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public float* GetAttnQBias(int layer) => _attnQBias[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public float* GetAttnKBias(int layer) => _attnKBias[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public float* GetAttnVBias(int layer) => _attnVBias[layer];

        public void Dispose() { }
    }
}
