using System;

namespace ChatNet.Core.Abstractions
{
    /// <summary>
    /// Core model interface for running forward passes on a loaded LLM.
    /// </summary>
    public interface IModel : IDisposable
    {
        ModelConfig Config { get; }

        /// <summary>
        /// Run a forward pass: given input token IDs, produce logits for next token.
        /// </summary>
        /// <param name="tokenIds">Input token sequence.</param>
        /// <param name="position">Starting position in the sequence (for KV cache).</param>
        /// <param name="logits">Output buffer for logits (size = vocab_size).</param>
        void Forward(ReadOnlySpan<int> tokenIds, int position, Span<float> logits);
    }
}
