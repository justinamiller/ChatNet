using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;

namespace ChatNet.Core.Models.Phi
{
    /// <summary>
    /// Weight tensor storage for Phi model.
    /// Supports both split gate/up and fused gate_up_proj tensor layouts.
    /// </summary>
    public sealed unsafe class PhiWeights : IDisposable
    {
        public GgmlType EmbeddingType { get; private set; }
        private byte* _embeddingPtr;
        private int _embeddingByteSize;

        private readonly byte*[] _attnNormWeight;
        private readonly byte*[] _ffnNormWeight;
        private readonly byte*[] _attnQWeight;
        private readonly byte*[] _attnKWeight;
        private readonly byte*[] _attnVWeight;
        private readonly byte*[] _attnOutputWeight;
        private readonly byte*[] _ffnGateWeight;
        private readonly byte*[] _ffnUpWeight;
        private readonly byte*[] _ffnDownWeight;

        // Fused gate_up for Phi-3
        private readonly byte*[] _ffnGateUpWeight;
        private readonly bool _hasFusedGateUp;

        public GgmlType[] AttnQType { get; }
        public GgmlType[] AttnKType { get; }
        public GgmlType[] AttnVType { get; }
        public GgmlType[] AttnOutputType { get; }
        public GgmlType[] FfnGateType { get; }
        public GgmlType[] FfnUpType { get; }
        public GgmlType[] FfnDownType { get; }
        public GgmlType[] FfnGateUpType { get; }

        private readonly int[] _attnNormSize;
        private readonly int[] _ffnNormSize;

        public GgmlType OutputType { get; private set; }
        private byte* _outputWeight;
        private byte* _finalNormWeight;
        private int _finalNormSize;

        public bool HasFusedGateUp => _hasFusedGateUp;

        public PhiWeights(MemoryMappedWeights weights, PhiConfig config)
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
            _ffnGateUpWeight = new byte*[layers];

            AttnQType = new GgmlType[layers];
            AttnKType = new GgmlType[layers];
            AttnVType = new GgmlType[layers];
            AttnOutputType = new GgmlType[layers];
            FfnGateType = new GgmlType[layers];
            FfnUpType = new GgmlType[layers];
            FfnDownType = new GgmlType[layers];
            FfnGateUpType = new GgmlType[layers];

            _attnNormSize = new int[layers];
            _ffnNormSize = new int[layers];

            _hasFusedGateUp = ResolveAll(weights, config);
        }

        private bool ResolveAll(MemoryMappedWeights w, PhiConfig config)
        {
            int layers = config.LayerCount;
            bool hasFusedGateUp = false;

            GgufTensorInfo embInfo = w.GetTensorInfo(PhiTensorNames.Embedding);
            EmbeddingType = embInfo.Type;
            _embeddingPtr = w.GetTensorPointer(PhiTensorNames.Embedding);
            _embeddingByteSize = (int)embInfo.ByteSize;

            if (w.HasTensor(PhiTensorNames.Output))
            {
                OutputType = w.GetTensorInfo(PhiTensorNames.Output).Type;
                _outputWeight = w.GetTensorPointer(PhiTensorNames.Output);
            }
            else
            {
                OutputType = EmbeddingType;
                _outputWeight = _embeddingPtr;
            }

            _finalNormWeight = w.GetTensorPointer(PhiTensorNames.OutputNorm);
            _finalNormSize = (int)w.GetTensorInfo(PhiTensorNames.OutputNorm).ByteSize;

            for (int l = 0; l < layers; l++)
            {
                string prefix = PhiTensorNames.BlockPrefix + l.ToString();

                string attnNormName = prefix + PhiTensorNames.AttnNormSuffix;
                _attnNormWeight[l] = w.GetTensorPointer(attnNormName);
                _attnNormSize[l] = (int)w.GetTensorInfo(attnNormName).ByteSize;

                string ffnNormName = prefix + PhiTensorNames.FfnNormSuffix;
                _ffnNormWeight[l] = w.GetTensorPointer(ffnNormName);
                _ffnNormSize[l] = (int)w.GetTensorInfo(ffnNormName).ByteSize;

                string aqName = prefix + PhiTensorNames.AttnQSuffix;
                _attnQWeight[l] = w.GetTensorPointer(aqName);
                AttnQType[l] = w.GetTensorInfo(aqName).Type;

                string akName = prefix + PhiTensorNames.AttnKSuffix;
                _attnKWeight[l] = w.GetTensorPointer(akName);
                AttnKType[l] = w.GetTensorInfo(akName).Type;

                string avName = prefix + PhiTensorNames.AttnVSuffix;
                _attnVWeight[l] = w.GetTensorPointer(avName);
                AttnVType[l] = w.GetTensorInfo(avName).Type;

                string aoName = prefix + PhiTensorNames.AttnOutputSuffix;
                _attnOutputWeight[l] = w.GetTensorPointer(aoName);
                AttnOutputType[l] = w.GetTensorInfo(aoName).Type;

                string fdName = prefix + PhiTensorNames.FfnDownSuffix;
                _ffnDownWeight[l] = w.GetTensorPointer(fdName);
                FfnDownType[l] = w.GetTensorInfo(fdName).Type;

                // Check for fused gate_up pattern first
                string fguName = prefix + PhiTensorNames.FfnGateUpSuffix;
                if (w.HasTensor(fguName))
                {
                    _ffnGateUpWeight[l] = w.GetTensorPointer(fguName);
                    FfnGateUpType[l] = w.GetTensorInfo(fguName).Type;
                    hasFusedGateUp = true;
                }
                else
                {
                    // Use split gate and up
                    string fgName = prefix + PhiTensorNames.FfnGateSuffix;
                    _ffnGateWeight[l] = w.GetTensorPointer(fgName);
                    FfnGateType[l] = w.GetTensorInfo(fgName).Type;

                    string fuName = prefix + PhiTensorNames.FfnUpSuffix;
                    _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                    FfnUpType[l] = w.GetTensorInfo(fuName).Type;
                }
            }

            return hasFusedGateUp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetEmbeddingData() => new ReadOnlySpan<byte>(_embeddingPtr, _embeddingByteSize);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetAttnNormWeight(int layer) => new ReadOnlySpan<byte>(_attnNormWeight[layer], _attnNormSize[layer]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetFfnNormWeight(int layer) => new ReadOnlySpan<byte>(_ffnNormWeight[layer], _ffnNormSize[layer]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetFinalNormWeight() => new ReadOnlySpan<byte>(_finalNormWeight, _finalNormSize);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnQWeight(int layer) => _attnQWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnKWeight(int layer) => _attnKWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnVWeight(int layer) => _attnVWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnOutputWeight(int layer) => _attnOutputWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnGateWeight(int layer) => _ffnGateWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnUpWeight(int layer) => _ffnUpWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnDownWeight(int layer) => _ffnDownWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnGateUpWeight(int layer) => _ffnGateUpWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetOutputWeight() => _outputWeight;

        public void Dispose() { }
    }
}
