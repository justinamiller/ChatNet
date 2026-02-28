using System;

namespace ChatNet.Core.Abstractions
{
    /// <summary>
    /// Sampling strategy abstraction for selecting the next token from logits.
    /// </summary>
    public interface ISampler
    {
        /// <summary>Given logits, return the selected token ID.</summary>
        int Sample(ReadOnlySpan<float> logits);
    }
}
