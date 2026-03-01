namespace ChatNet.Core.Models.Phi
{
    /// <summary>
    /// Tensor name constants for Phi architecture in GGUF format.
    /// Supports both split gate/up and fused gate_up_proj patterns.
    /// </summary>
    internal static class PhiTensorNames
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

        // Phi-3 may use a fused gate_up projection
        public const string FfnGateUpSuffix = ".ffn_gate_up.weight";

        public const string BlockPrefix = "blk.";

        public static string LayerName(int layer, string suffix)
        {
            return BlockPrefix + layer.ToString() + suffix;
        }
    }
}
