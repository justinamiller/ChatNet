using System;
using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Models.Gemma
{
    /// <summary>
    /// Gemma-specific hyperparameters derived from ModelConfig.
    /// Key differences from Llama: GELU activation, embedding scaling, norm weight +1 offset.
    /// Gemma 2 adds: post-norms, attention logit soft-capping, final logit soft-capping.
    /// </summary>
    public sealed class GemmaConfig
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
        public float EmbeddingScale { get; }

        // Gemma 2 soft-capping (0 = disabled)
        public float AttnLogitSoftcap { get; }
        public float FinalLogitSoftcap { get; }

        // Gemma 2 has post-norms (detected from weights)
        public bool HasPostNorms { get; set; }

        public GemmaConfig(ModelConfig config)
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
            EmbeddingScale = MathF.Sqrt(Dim);
            AttnLogitSoftcap = config.AttnLogitSoftcap;
            FinalLogitSoftcap = config.FinalLogitSoftcap;
        }
    }
}
