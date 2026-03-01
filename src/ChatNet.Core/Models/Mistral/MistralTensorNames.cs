namespace ChatNet.Core.Models.Mistral
{
    /// <summary>
    /// Tensor name constants for Mistral architecture in GGUF format.
    /// Mistral uses the same standard GGUF tensor names as Llama.
    /// </summary>
    internal static class MistralTensorNames
    {
        public const string Embedding = "token_embd.weight";
        public const string OutputNorm = "output_norm.weight";
        public const string Output = "output.weight";

        public const string AttnNormSuffix = ".attn_norm.weight";
        public const string AttnQSuffix = ".attn_q.weight";
        public const string AttnKSuffix = ".attn_k.weight";
        public const string AttnVSuffix = ".attn_v.weight";
        public const string AttnOutputSuffix = ".attn_output.weight";

        public const string FfnNormSuffix = ".ffn_norm.weight";
        public const string FfnGateSuffix = ".ffn_gate.weight";
        public const string FfnUpSuffix = ".ffn_up.weight";
        public const string FfnDownSuffix = ".ffn_down.weight";

        public const string BlockPrefix = "blk.";

        public static string LayerName(int layer, string suffix)
        {
            return BlockPrefix + layer.ToString() + suffix;
        }
    }
}
