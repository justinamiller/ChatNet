using System;
using System.Collections.Generic;

namespace ChatNet.Core.Tokenizer
{
    /// <summary>
    /// Vocabulary storage loaded from GGUF metadata.
    /// Stores token strings and their scores for BPE merging.
    /// </summary>
    public sealed class TokenVocab
    {
        public string[] Tokens { get; }
        public float[] Scores { get; }
        public int[]? TokenTypes { get; }
        public int Size { get; }

        // Reverse lookup: token string -> token ID (built at load time, not hot path)
        private readonly Dictionary<string, int> _tokenToId;

        public TokenVocab(string[] tokens, float[] scores, int[]? tokenTypes)
        {
            Tokens = tokens;
            Scores = scores;
            TokenTypes = tokenTypes;
            Size = tokens.Length;

            _tokenToId = new Dictionary<string, int>(tokens.Length);
            for (int i = 0; i < tokens.Length; i++)
            {
                // Some tokens may be duplicates; keep the first one
                if (!_tokenToId.ContainsKey(tokens[i]))
                {
                    _tokenToId[tokens[i]] = i;
                }
            }
        }

        /// <summary>Look up token ID by string. Returns -1 if not found.</summary>
        public int GetTokenId(string token)
        {
            if (_tokenToId.TryGetValue(token, out int id))
            {
                return id;
            }
            return -1;
        }

        /// <summary>Check if a token string exists in vocab.</summary>
        public bool Contains(string token)
        {
            return _tokenToId.ContainsKey(token);
        }
    }
}
