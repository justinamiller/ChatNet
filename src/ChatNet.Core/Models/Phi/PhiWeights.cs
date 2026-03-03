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

        /// <summary>
        /// Actual FFN intermediate dimension derived from the ffn_down tensor's input dimension (ne[0]).
        /// This is the ground truth from the tensor data, independent of GGUF metadata.
        /// </summary>
        public int ActualFfnHiddenDim { get; private set; }

        /// <summary>
        /// Actual fused QKV output dimension derived from the tensor's ne[1].
        /// For MHA (Phi-3-mini): should be 3*dim. For GQA: dim + 2*kvDim.
        /// </summary>
        public int ActualQkvOutDim { get; private set; }

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

            // Debug: report dimension derivation and splitting results
            {
                int dim = config.Dim;
                int kvDim = config.KvDim;
                Console.Error.WriteLine("[DEBUG] PhiWeights: actualFfnHiddenDim=" + ActualFfnHiddenDim +
                    " configHiddenDim=" + config.HiddenDim +
                    (ActualFfnHiddenDim != config.HiddenDim ? " MISMATCH" : " OK"));

                string qkvName0 = PhiTensorNames.BlockPrefix + "0" + PhiTensorNames.AttnQkvSuffix;
                if (weights.HasTensor(qkvName0))
                {
                    var info = weights.GetTensorInfo(qkvName0);
                    Console.Error.WriteLine("[DEBUG] PhiWeights: Split fused QKV dims=[" +
                        info.Dimensions[0] + "," + info.Dimensions[1] + "]" +
                        " into Q[" + dim + "], K[" + kvDim + "], V[" + kvDim + "]" +
                        " type=" + info.Type);
                }

                string fguName0 = PhiTensorNames.BlockPrefix + "0" + PhiTensorNames.FfnGateUpSuffix;
                string fuName0 = PhiTensorNames.BlockPrefix + "0" + PhiTensorNames.FfnUpSuffix;
                string guName0 = weights.HasTensor(fguName0) ? fguName0 : fuName0;
                if (weights.HasTensor(guName0))
                {
                    var info = weights.GetTensorInfo(guName0);
                    if ((int)info.Dimensions[1] >= ActualFfnHiddenDim * 2)
                    {
                        Console.Error.WriteLine("[DEBUG] PhiWeights: Split fused GateUp '" + guName0 +
                            "' dims=[" + info.Dimensions[0] + "," + info.Dimensions[1] + "]" +
                            " into gate[" + ActualFfnHiddenDim + "], up[" + ActualFfnHiddenDim + "]" +
                            " type=" + info.Type);
                    }
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

                // Check for fused QKV (Phi-3 pattern) — split into separate Q/K/V pointers
                string qkvName = prefix + PhiTensorNames.AttnQkvSuffix;
                if (w.HasTensor(qkvName))
                {
                    var qkvInfo = w.GetTensorInfo(qkvName);
                    GgmlType qkvType = qkvInfo.Type;
                    byte* qkvPtr = w.GetTensorPointer(qkvName);
                    long qkvBytesPerRow = (long)qkvInfo.ByteSize / (long)qkvInfo.Dimensions[1];
                    int splitDim = config.Dim;
                    int splitKvDim = config.KvDim;

                    _attnQWeight[l] = qkvPtr;
                    _attnKWeight[l] = qkvPtr + (long)splitDim * qkvBytesPerRow;
                    _attnVWeight[l] = qkvPtr + (long)(splitDim + splitKvDim) * qkvBytesPerRow;
                    AttnQType[l] = qkvType;
                    AttnKType[l] = qkvType;
                    AttnVType[l] = qkvType;
                }
                else
                {
                    // Separate Q/K/V tensors
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

                // Derive actual FFN hidden dim from ffn_down tensor (ne[0] = intermediate_size)
                int actualHiddenDim = (int)w.GetTensorInfo(fdName).Dimensions[0];
                if (l == 0) ActualFfnHiddenDim = actualHiddenDim;

                // Resolve gate/up projections — split fused tensors into separate pointers
                string fguName = prefix + PhiTensorNames.FfnGateUpSuffix;
                string fuName = prefix + PhiTensorNames.FfnUpSuffix;
                string fgName = prefix + PhiTensorNames.FfnGateSuffix;

                if (w.HasTensor(fgName))
                {
                    // Separate gate and up tensors
                    _ffnGateWeight[l] = w.GetTensorPointer(fgName);
                    FfnGateType[l] = w.GetTensorInfo(fgName).Type;
                    _ffnUpWeight[l] = w.GetTensorPointer(fuName);
                    FfnUpType[l] = w.GetTensorInfo(fuName).Type;
                }
                else
                {
                    // Fused gate_up tensor — split into separate gate and up sub-pointers
                    string guTensorName = w.HasTensor(fguName) ? fguName : fuName;
                    var guInfo = w.GetTensorInfo(guTensorName);
                    int guOutDim = (int)guInfo.Dimensions[1];
                    GgmlType guType = guInfo.Type;
                    byte* guPtr = w.GetTensorPointer(guTensorName);

                    if (guOutDim >= actualHiddenDim * 2)
                    {
                        // Fused: first half rows = gate, second half rows = up
                        long guBytesPerRow = (long)guInfo.ByteSize / (long)guOutDim;
                        _ffnGateWeight[l] = guPtr;
                        _ffnUpWeight[l] = guPtr + (long)actualHiddenDim * guBytesPerRow;
                        FfnGateType[l] = guType;
                        FfnUpType[l] = guType;
                    }
                    else
                    {
                        // Non-fused up tensor (no gate found)
                        _ffnUpWeight[l] = guPtr;
                        FfnUpType[l] = guType;
                        if (l == 0)
                        {
                            Console.Error.WriteLine("[WARN] PhiWeights: No ffn_gate tensor and ffn_up is not fused (dims=[" +
                                guInfo.Dimensions[0] + "," + guOutDim + "]). Model output may be incorrect.");
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
