using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;

namespace ChatNet.Core.Models.Phi
{
    /// <summary>
    /// Weight tensor storage for Phi model.
    /// Supports fused QKV (attn_qkv.weight) and fused gate_up projection layouts.
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

        // Fused QKV for Phi-3
        private readonly byte*[] _attnQkvWeight;
        private bool _hasFusedQkv;

        // Fused gate_up for Phi-3
        private readonly byte*[] _ffnGateUpWeight;
        private bool _hasFusedGateUp;

        public GgmlType[] AttnQType { get; }
        public GgmlType[] AttnKType { get; }
        public GgmlType[] AttnVType { get; }
        public GgmlType[] AttnOutputType { get; }
        public GgmlType[] AttnQkvType { get; }
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

        public bool HasFusedQkv => _hasFusedQkv;
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
            _attnQkvWeight = new byte*[layers];
            _ffnGateWeight = new byte*[layers];
            _ffnUpWeight = new byte*[layers];
            _ffnDownWeight = new byte*[layers];
            _ffnGateUpWeight = new byte*[layers];

            AttnQType = new GgmlType[layers];
            AttnKType = new GgmlType[layers];
            AttnVType = new GgmlType[layers];
            AttnOutputType = new GgmlType[layers];
            AttnQkvType = new GgmlType[layers];
            FfnGateType = new GgmlType[layers];
            FfnUpType = new GgmlType[layers];
            FfnDownType = new GgmlType[layers];
            FfnGateUpType = new GgmlType[layers];

            _attnNormSize = new int[layers];
            _ffnNormSize = new int[layers];

            ResolveAll(weights, config);

            // Debug: verify fused tensor dimensions for first layer
            if (_hasFusedQkv || _hasFusedGateUp)
            {
                int dim = config.Dim;
                int kvDim = config.KvDim;
                int hiddenDim = config.HiddenDim;

                if (_hasFusedQkv)
                {
                    string qkvName = PhiTensorNames.BlockPrefix + "0" + PhiTensorNames.AttnQkvSuffix;
                    if (weights.HasTensor(qkvName))
                    {
                        var info = weights.GetTensorInfo(qkvName);
                        ulong expectedOut = (ulong)(dim + kvDim + kvDim);
                        Console.Error.WriteLine("[DEBUG] PhiWeights: QKV tensor dims=[" +
                            info.Dimensions[0] + "," + info.Dimensions[1] + "]" +
                            " expected=[" + dim + "," + expectedOut + "]" +
                            " type=" + info.Type);
                    }
                }

                if (_hasFusedGateUp)
                {
                    string prefix0 = PhiTensorNames.BlockPrefix + "0";
                    string fguName = prefix0 + PhiTensorNames.FfnGateUpSuffix;
                    string fuName = prefix0 + PhiTensorNames.FfnUpSuffix;
                    string tensorName = weights.HasTensor(fguName) ? fguName : fuName;
                    var info = weights.GetTensorInfo(tensorName);
                    ulong expectedOut = (ulong)(hiddenDim * 2);
                    Console.Error.WriteLine("[DEBUG] PhiWeights: GateUp tensor '" + tensorName +
                        "' dims=[" + info.Dimensions[0] + "," + info.Dimensions[1] + "]" +
                        " expected=[" + dim + "," + expectedOut + "]" +
                        " type=" + info.Type);
                }
            }
        }

        private void ResolveAll(MemoryMappedWeights w, PhiConfig config)
        {
            int layers = config.LayerCount;

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

                // Check for fused QKV first (Phi-3 pattern)
                string qkvName = prefix + PhiTensorNames.AttnQkvSuffix;
                if (w.HasTensor(qkvName))
                {
                    _attnQkvWeight[l] = w.GetTensorPointer(qkvName);
                    AttnQkvType[l] = w.GetTensorInfo(qkvName).Type;
                    _hasFusedQkv = true;
                }
                else
                {
                    // Fallback to separate Q/K/V
                    string aqName = prefix + PhiTensorNames.AttnQSuffix;
                    _attnQWeight[l] = w.GetTensorPointer(aqName);
                    AttnQType[l] = w.GetTensorInfo(aqName).Type;

                    string akName = prefix + PhiTensorNames.AttnKSuffix;
                    _attnKWeight[l] = w.GetTensorPointer(akName);
                    AttnKType[l] = w.GetTensorInfo(akName).Type;

                    string avName = prefix + PhiTensorNames.AttnVSuffix;
                    _attnVWeight[l] = w.GetTensorPointer(avName);
                    AttnVType[l] = w.GetTensorInfo(avName).Type;
                }

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
                    _hasFusedGateUp = true;
                }
                else
                {
                    // Check if ffn_up is actually fused gate_up (Phi-3 sometimes stores fused as ffn_up)
                    string fuName = prefix + PhiTensorNames.FfnUpSuffix;
                    string fgName = prefix + PhiTensorNames.FfnGateSuffix;
                    if (w.HasTensor(fgName))
                    {
                        _ffnGateWeight[l] = w.GetTensorPointer(fgName);
                        FfnGateType[l] = w.GetTensorInfo(fgName).Type;
                        _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                        FfnUpType[l] = w.GetTensorInfo(fuName).Type;
                    }
                    else
                    {
                        // Only ffn_up exists - verify dimensions to determine if fused gate_up
                        var fuInfo = w.GetTensorInfo(fuName);
                        long actualOutDim = fuInfo.NDimensions >= 2 ? (long)fuInfo.Dimensions[1] : 0;
                        long expectedFusedOut = (long)config.HiddenDim * 2;
                        long expectedSeparateOut = (long)config.HiddenDim;

                        if (actualOutDim == expectedFusedOut)
                        {
                            // Dimensions confirm fused gate_up: [dim, 2*hiddenDim]
                            _ffnGateUpWeight[l] = w.GetTensorPointer(fuName);
                            FfnGateUpType[l] = fuInfo.Type;
                            _hasFusedGateUp = true;
                        }
                        else if (actualOutDim == expectedSeparateOut)
                        {
                            // Dimensions show non-fused up: [dim, hiddenDim]
                            // This means ffn_up is a regular up projection, not fused.
                            // Without a gate tensor, SiLU gating won't work correctly.
                            // Treat as fused anyway but warn loudly.
                            Console.Error.WriteLine("[WARN] PhiWeights layer " + l +
                                ": ffn_up.weight has non-fused dims=[" +
                                fuInfo.Dimensions[0] + "," + actualOutDim +
                                "] (expected fused [" + config.Dim + "," + expectedFusedOut +
                                "]). No ffn_gate tensor found either. Model output will be incorrect.");
                            _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                            FfnUpType[l] = fuInfo.Type;
                        }
                        else
                        {
                            // Unknown dimension - treat as fused and warn
                            Console.Error.WriteLine("[WARN] PhiWeights layer " + l +
                                ": ffn_up.weight has unexpected dims=[" +
                                fuInfo.Dimensions[0] + "," + actualOutDim +
                                "] (expected fused=" + expectedFusedOut +
                                " or separate=" + expectedSeparateOut + ")");
                            _ffnGateUpWeight[l] = w.GetTensorPointer(fuName);
                            FfnGateUpType[l] = fuInfo.Type;
                            _hasFusedGateUp = true;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetEmbeddingData() => new ReadOnlySpan<byte>(_embeddingPtr, _embeddingByteSize);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetAttnNormWeight(int layer) => new ReadOnlySpan<byte>(_attnNormWeight[layer], _attnNormSize[layer]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetFfnNormWeight(int layer) => new ReadOnlySpan<byte>(_ffnNormWeight[layer], _ffnNormSize[layer]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public ReadOnlySpan<byte> GetFinalNormWeight() => new ReadOnlySpan<byte>(_finalNormWeight, _finalNormSize);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnQWeight(int layer) => _attnQWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnKWeight(int layer) => _attnKWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnVWeight(int layer) => _attnVWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnQkvWeight(int layer) => _attnQkvWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetAttnOutputWeight(int layer) => _attnOutputWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnGateWeight(int layer) => _ffnGateWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnUpWeight(int layer) => _ffnUpWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnDownWeight(int layer) => _ffnDownWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetFfnGateUpWeight(int layer) => _ffnGateUpWeight[layer];
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public byte* GetOutputWeight() => _outputWeight;

        public void Dispose() { }
    }
}
