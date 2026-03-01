using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;

namespace ChatNet.Core.Models.Gemma
{
    /// <summary>
    /// Weight tensor storage for Gemma model.
    /// Handles the case where output.weight may not exist (tied to embeddings).
    /// </summary>
    public sealed unsafe class GemmaWeights : IDisposable
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

        public GgmlType[] AttnQType { get; }
        public GgmlType[] AttnKType { get; }
        public GgmlType[] AttnVType { get; }
        public GgmlType[] AttnOutputType { get; }
        public GgmlType[] FfnGateType { get; }
        public GgmlType[] FfnUpType { get; }
        public GgmlType[] FfnDownType { get; }

        private readonly int[] _attnNormSize;
        private readonly int[] _ffnNormSize;

        public GgmlType OutputType { get; private set; }
        private byte* _outputWeight;
        private byte* _finalNormWeight;
        private int _finalNormSize;
        public bool TiedEmbeddings { get; private set; }

        public GemmaWeights(MemoryMappedWeights weights, GemmaConfig config)
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

            AttnQType = new GgmlType[layers];
            AttnKType = new GgmlType[layers];
            AttnVType = new GgmlType[layers];
            AttnOutputType = new GgmlType[layers];
            FfnGateType = new GgmlType[layers];
            FfnUpType = new GgmlType[layers];
            FfnDownType = new GgmlType[layers];

            _attnNormSize = new int[layers];
            _ffnNormSize = new int[layers];

            ResolveAll(weights, config);
        }

        private void ResolveAll(MemoryMappedWeights w, GemmaConfig config)
        {
            int layers = config.LayerCount;

            GgufTensorInfo embInfo = w.GetTensorInfo(GemmaTensorNames.Embedding);
            EmbeddingType = embInfo.Type;
            _embeddingPtr = w.GetTensorPointer(GemmaTensorNames.Embedding);
            _embeddingByteSize = (int)embInfo.ByteSize;

            // Gemma often ties output to embeddings
            if (w.HasTensor(GemmaTensorNames.Output))
            {
                OutputType = w.GetTensorInfo(GemmaTensorNames.Output).Type;
                _outputWeight = w.GetTensorPointer(GemmaTensorNames.Output);
                TiedEmbeddings = false;
            }
            else
            {
                OutputType = EmbeddingType;
                _outputWeight = _embeddingPtr;
                TiedEmbeddings = true;
            }

            _finalNormWeight = w.GetTensorPointer(GemmaTensorNames.OutputNorm);
            _finalNormSize = (int)w.GetTensorInfo(GemmaTensorNames.OutputNorm).ByteSize;

            for (int l = 0; l < layers; l++)
            {
                string prefix = GemmaTensorNames.BlockPrefix + l.ToString();

                string attnNormName = prefix + GemmaTensorNames.AttnNormSuffix;
                _attnNormWeight[l] = w.GetTensorPointer(attnNormName);
                _attnNormSize[l] = (int)w.GetTensorInfo(attnNormName).ByteSize;

                string ffnNormName = prefix + GemmaTensorNames.FfnNormSuffix;
                _ffnNormWeight[l] = w.GetTensorPointer(ffnNormName);
                _ffnNormSize[l] = (int)w.GetTensorInfo(ffnNormName).ByteSize;

                string aqName = prefix + GemmaTensorNames.AttnQSuffix;
                _attnQWeight[l] = w.GetTensorPointer(aqName);
                AttnQType[l] = w.GetTensorInfo(aqName).Type;

                string akName = prefix + GemmaTensorNames.AttnKSuffix;
                _attnKWeight[l] = w.GetTensorPointer(akName);
                AttnKType[l] = w.GetTensorInfo(akName).Type;

                string avName = prefix + GemmaTensorNames.AttnVSuffix;
                _attnVWeight[l] = w.GetTensorPointer(avName);
                AttnVType[l] = w.GetTensorInfo(avName).Type;

                string aoName = prefix + GemmaTensorNames.AttnOutputSuffix;
                _attnOutputWeight[l] = w.GetTensorPointer(aoName);
                AttnOutputType[l] = w.GetTensorInfo(aoName).Type;

                string fgName = prefix + GemmaTensorNames.FfnGateSuffix;
                _ffnGateWeight[l] = w.GetTensorPointer(fgName);
                FfnGateType[l] = w.GetTensorInfo(fgName).Type;

                string fuName = prefix + GemmaTensorNames.FfnUpSuffix;
                _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                FfnUpType[l] = w.GetTensorInfo(fuName).Type;

                string fdName = prefix + GemmaTensorNames.FfnDownSuffix;
                _ffnDownWeight[l] = w.GetTensorPointer(fdName);
                FfnDownType[l] = w.GetTensorInfo(fdName).Type;
            }
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetOutputWeight() => _outputWeight;

        public void Dispose() { }
    }
}
