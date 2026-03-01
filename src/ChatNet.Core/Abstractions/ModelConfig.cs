namespace ChatNet.Core.Abstractions
{
    /// <summary>
    /// Unified model configuration extracted from metadata.
    /// </summary>
    public sealed class ModelConfig
    {
        public string Architecture { get; set; } = "";
        public string ModelName { get; set; } = "";
        public int VocabSize { get; set; }
        public int EmbeddingDim { get; set; }
        public int LayerCount { get; set; }
        public int AttentionHeadCount { get; set; }
        public int KeyValueHeadCount { get; set; }
        public int HeadDim { get; set; }
        public int FeedForwardDim { get; set; }
        public int ContextLength { get; set; }
        public float RopeFreqBase { get; set; } = 10000.0f;
        public float RmsNormEpsilon { get; set; } = 1e-5f;
        public int BosTokenId { get; set; } = 1;
        public int EosTokenId { get; set; } = 2;

        // Gemma 2 soft-capping parameters (0 = disabled)
        public float AttnLogitSoftcap { get; set; }
        public float FinalLogitSoftcap { get; set; }
    }
}
