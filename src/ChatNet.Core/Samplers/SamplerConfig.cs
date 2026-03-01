namespace ChatNet.Core.Samplers
{
    /// <summary>
    /// Configuration for sampling parameters.
    /// </summary>
    public sealed class SamplerConfig
    {
        public float Temperature { get; set; } = 0.0f;
        public int TopK { get; set; } = 40;
        public float TopP { get; set; } = 0.9f;
        public float RepetitionPenalty { get; set; } = 1.1f;
    }
}
