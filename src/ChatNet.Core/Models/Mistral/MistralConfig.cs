using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Models.Mistral
{
    /// <summary>
    /// Mistral-specific hyperparameters derived from ModelConfig.
    /// Mistral is architecturally similar to Llama with optional sliding window attention.
    /// </summary>
    public sealed class MistralConfig
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

        public MistralConfig(ModelConfig config)
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
