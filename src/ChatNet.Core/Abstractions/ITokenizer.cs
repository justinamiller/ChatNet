using System;

namespace ChatNet.Core.Abstractions
{
    /// <summary>
    /// Tokenizer abstraction for encoding text to token IDs and decoding back.
    /// </summary>
    public interface ITokenizer
    {
        int VocabSize { get; }
        int BosToken { get; }
        int EosToken { get; }

        /// <summary>Encode text to token IDs. Returns number of tokens written.</summary>
        int Encode(ReadOnlySpan<char> text, Span<int> outputTokens);

        /// <summary>Decode a single token ID to its string representation.</summary>
        string Decode(int tokenId);
    }
}
