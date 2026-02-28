using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;

namespace ChatNet.Core.Models.Llama
{
    /// <summary>
    /// Weight tensor storage for Llama model.
    /// All pointers are resolved at construction time to avoid string allocation in the forward pass.
    /// </summary>
    public sealed unsafe class LlamaWeights : IDisposable
    {
        // Embedding
        public GgmlType EmbeddingType { get; private set; }
        private byte* _embeddingPtr;
        private int _embeddingByteSize;

        // Per-layer cached pointers (resolved at construction, zero alloc in forward pass)
        private readonly byte*[] _attnNormWeight;
        private readonly byte*[] _ffnNormWeight;
        private readonly byte*[] _attnQWeight;
        private readonly byte*[] _attnKWeight;
        private readonly byte*[] _attnVWeight;
        private readonly byte*[] _attnOutputWeight;
        private readonly byte*[] _ffnGateWeight;
        private readonly byte*[] _ffnUpWeight;
        private readonly byte*[] _ffnDownWeight;

        // Per-layer quantization types
        public GgmlType[] AttnQType { get; }
        public GgmlType[] AttnKType { get; }
        public GgmlType[] AttnVType { get; }
        public GgmlType[] AttnOutputType { get; }
        public GgmlType[] FfnGateType { get; }
        public GgmlType[] FfnUpType { get; }
        public GgmlType[] FfnDownType { get; }

        // Per-layer norm data sizes (for span construction)
        private readonly int[] _attnNormSize;
        private readonly int[] _ffnNormSize;

        // Output projection
        public GgmlType OutputType { get; private set; }
        private byte* _outputWeight;
        private byte* _finalNormWeight;
        private int _finalNormSize;

        public LlamaWeights(MemoryMappedWeights weights, LlamaConfig config)
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

        private void ResolveAll(MemoryMappedWeights w, LlamaConfig config)
        {
            int layers = config.LayerCount;

            // Embedding
            GgufTensorInfo embInfo = w.GetTensorInfo("token_embd.weight");
            EmbeddingType = embInfo.Type;
            _embeddingPtr = w.GetTensorPointer("token_embd.weight");
            _embeddingByteSize = (int)embInfo.ByteSize;

            // Output weight
            if (w.HasTensor("output.weight"))
            {
                OutputType = w.GetTensorInfo("output.weight").Type;
                _outputWeight = w.GetTensorPointer("output.weight");
            }
            else
            {
                OutputType = EmbeddingType;
                _outputWeight = _embeddingPtr;
            }

            // Final norm
            _finalNormWeight = w.GetTensorPointer("output_norm.weight");
            _finalNormSize = (int)w.GetTensorInfo("output_norm.weight").ByteSize;

            // Per-layer weights
            for (int l = 0; l < layers; l++)
            {
                string prefix = "blk." + l.ToString();

                string attnNormName = prefix + ".attn_norm.weight";
                _attnNormWeight[l] = w.GetTensorPointer(attnNormName);
                _attnNormSize[l] = (int)w.GetTensorInfo(attnNormName).ByteSize;

                string ffnNormName = prefix + ".ffn_norm.weight";
                _ffnNormWeight[l] = w.GetTensorPointer(ffnNormName);
                _ffnNormSize[l] = (int)w.GetTensorInfo(ffnNormName).ByteSize;

                string aqName = prefix + ".attn_q.weight";
                _attnQWeight[l] = w.GetTensorPointer(aqName);
                AttnQType[l] = w.GetTensorInfo(aqName).Type;

                string akName = prefix + ".attn_k.weight";
                _attnKWeight[l] = w.GetTensorPointer(akName);
                AttnKType[l] = w.GetTensorInfo(akName).Type;

                string avName = prefix + ".attn_v.weight";
                _attnVWeight[l] = w.GetTensorPointer(avName);
                AttnVType[l] = w.GetTensorInfo(avName).Type;

                string aoName = prefix + ".attn_output.weight";
                _attnOutputWeight[l] = w.GetTensorPointer(aoName);
                AttnOutputType[l] = w.GetTensorInfo(aoName).Type;

                string fgName = prefix + ".ffn_gate.weight";
                _ffnGateWeight[l] = w.GetTensorPointer(fgName);
                FfnGateType[l] = w.GetTensorInfo(fgName).Type;

                string fuName = prefix + ".ffn_up.weight";
                _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                FfnUpType[l] = w.GetTensorInfo(fuName).Type;

                string fdName = prefix + ".ffn_down.weight";
                _ffnDownWeight[l] = w.GetTensorPointer(fdName);
                FfnDownType[l] = w.GetTensorInfo(fdName).Type;
            }
        }

        // All accessors return pre-cached pointers - zero allocation in forward pass

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetEmbeddingData()
        {
            return new ReadOnlySpan<byte>(_embeddingPtr, _embeddingByteSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetAttnNormWeight(int layer)
        {
            return new ReadOnlySpan<byte>(_attnNormWeight[layer], _attnNormSize[layer]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetFfnNormWeight(int layer)
        {
            return new ReadOnlySpan<byte>(_ffnNormWeight[layer], _ffnNormSize[layer]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<byte> GetFinalNormWeight()
        {
            return new ReadOnlySpan<byte>(_finalNormWeight, _finalNormSize);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetAttnQWeight(int layer) => _attnQWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetAttnKWeight(int layer) => _attnKWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetAttnVWeight(int layer) => _attnVWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetAttnOutputWeight(int layer) => _attnOutputWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetFfnGateWeight(int layer) => _ffnGateWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetFfnUpWeight(int layer) => _ffnUpWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetFfnDownWeight(int layer) => _ffnDownWeight[layer];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte* GetOutputWeight() => _outputWeight;

        public void Dispose()
        {
            // MemoryMappedWeights is disposed separately by the engine
        }
    }
}
