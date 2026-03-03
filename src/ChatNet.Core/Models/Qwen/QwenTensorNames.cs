namespace ChatNet.Core.Models.Qwen
{
    /// <summary>
    /// Tensor name constants for Qwen2 architecture in GGUF format.
    /// Qwen2 uses standard GGUF tensor names plus optional bias tensors.
    /// </summary>
    internal static class QwenTensorNames
    {
        // Global tensors
        public const string Embedding = "token_embd.weight";
        public const string OutputNorm = "output_norm.weight";
        public const string Output = "output.weight";

        // Per-layer attention
        public const string AttnNormSuffix = ".attn_norm.weight";
        public const string AttnQSuffix = ".attn_q.weight";
        public const string AttnKSuffix = ".attn_k.weight";
        public const string AttnVSuffix = ".attn_v.weight";
        public const string AttnOutputSuffix = ".attn_output.weight";

        // Per-layer attention biases (Qwen2 specific)
        public const string AttnQBiasSuffix = ".attn_q.bias";
        public const string AttnKBiasSuffix = ".attn_k.bias";
        public const string AttnVBiasSuffix = ".attn_v.bias";

        // Per-layer FFN
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
