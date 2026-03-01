using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Models.Phi
{
    /// <summary>
    /// Phi-specific hyperparameters derived from ModelConfig.
    /// Supports Phi-3 (Llama-like: RMSNorm, SiLU-gated FFN, RoPE).
    /// </summary>
    public sealed class PhiConfig
    {
        public int Dim { get; }
        public int HiddenDim { get; }
        public int LayerCount { get; }
        public int HeadCount { get; }
        public int KvHeadCount { get; }
        public int HeadDim { get; }
        public int KvDim { get; }
        public int VocabSize { get; }
        public int ContextLength { get; }
        public float RopeFreqBase { get; }
        public float RmsNormEps { get; }
        public int KvMul { get; }

        public PhiConfig(ModelConfig config)
        {
            Dim = config.EmbeddingDim;
            HiddenDim = config.FeedForwardDim;
            LayerCount = config.LayerCount;
            HeadCount = config.AttentionHeadCount;
            KvHeadCount = config.KeyValueHeadCount;
            HeadDim = config.HeadDim;
            KvDim = KvHeadCount * HeadDim;
            VocabSize = config.VocabSize;
            ContextLength = config.ContextLength;
            RopeFreqBase = config.RopeFreqBase;
            RmsNormEps = config.RmsNormEpsilon;
            KvMul = HeadCount / KvHeadCount;
        }
    }
}
