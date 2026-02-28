using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Gguf;

namespace ChatNet.Core.Tokenizer
{
    /// <summary>
    /// SentencePiece-style BPE tokenizer that reads vocabulary from GGUF metadata.
    /// Implements greedy BPE merge algorithm using scores from the vocabulary.
    /// </summary>
    public sealed class BpeTokenizer : ITokenizer
    {
        private readonly TokenVocab _vocab;
        private readonly int _bosToken;
        private readonly int _eosToken;

        // Byte fallback tokens: maps byte value -> token ID for "<0xHH>" tokens
        private readonly int[] _byteTokens;

        public int VocabSize => _vocab.Size;
        public int BosToken => _bosToken;
        public int EosToken => _eosToken;

        public BpeTokenizer(GgufMetadata metadata)
        {
            string[]? tokens = metadata.GetStringArray("tokenizer.ggml.tokens");
            if (tokens == null)
            {
                throw new InvalidOperationException("No tokenizer vocabulary found in GGUF metadata");
            }

            float[]? scores = metadata.GetFloatArray("tokenizer.ggml.scores");
            if (scores == null)
            {
                // Create default scores (index-based)
                scores = new float[tokens.Length];
                for (int i = 0; i < scores.Length; i++)
                {
                    scores[i] = -i;
                }
            }

            int[]? tokenTypes = metadata.GetIntArray("tokenizer.ggml.token_type");

            _vocab = new TokenVocab(tokens, scores, tokenTypes);
            _bosToken = metadata.GetInt32("tokenizer.ggml.bos_token_id", 1);
            _eosToken = metadata.GetInt32("tokenizer.ggml.eos_token_id", 2);

            // Build byte fallback mapping
            _byteTokens = new int[256];
            for (int i = 0; i < 256; i++)
            {
                _byteTokens[i] = -1;
            }

            for (int i = 0; i < tokens.Length; i++)
            {
                string tok = tokens[i];
                // Match "<0xHH>" pattern
                if (tok.Length == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>')
                {
                    if (TryParseHexByte(tok, 3, out byte byteVal))
                    {
                        _byteTokens[byteVal] = i;
                    }
                }
            }
        }

        /// <summary>
        /// Encode text to token IDs using SentencePiece-style BPE.
        /// Prepends BOS token. Returns number of tokens written.
        /// </summary>
        public int Encode(ReadOnlySpan<char> text, Span<int> outputTokens)
        {
            if (text.Length == 0)
            {
                outputTokens[0] = _bosToken;
                return 1;
            }

            // Convert text to UTF-8 bytes
            int maxBytes = Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[] utf8Bytes = new byte[maxBytes];
            int byteCount = Encoding.UTF8.GetBytes(text, utf8Bytes);

            // Initialize: each byte or recognized character as a separate token
            // First, try to match the SentencePiece style: prepend space (▁ = U+2581)
            // SentencePiece prepends ▁ to the beginning of the text
            string processedText = "\u2581" + text.ToString().Replace(" ", "\u2581");

            // Convert processed text to list of single-char strings for merging
            // Actually, we need to work with UTF-8 bytes and the vocabulary
            // Let's use a simpler approach: character-level initialization then BPE merge

            // Build initial token list from the processed text
            var symbols = new List<string>(processedText.Length);
            int charIdx = 0;
            while (charIdx < processedText.Length)
            {
                // Try to find the longest single-char token
                string ch = processedText.Substring(charIdx, 1);
                if (_vocab.Contains(ch))
                {
                    symbols.Add(ch);
                }
                else
                {
                    // Fallback to byte-level tokens
                    byte[] charBytes = Encoding.UTF8.GetBytes(ch);
                    for (int b = 0; b < charBytes.Length; b++)
                    {
                        string byteTok = $"<0x{charBytes[b]:X2}>";
                        symbols.Add(byteTok);
                    }
                }
                charIdx++;
            }

            // BPE merge loop: repeatedly find the best merge (highest score)
            bool merged = true;
            while (merged && symbols.Count > 1)
            {
                merged = false;
                float bestScore = float.NegativeInfinity;
                int bestIdx = -1;
                string bestMerge = "";

                // Find the adjacent pair with the highest score in vocabulary
                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    string candidate = symbols[i] + symbols[i + 1];
                    int tokenId = _vocab.GetTokenId(candidate);
                    if (tokenId >= 0)
                    {
                        float score = _vocab.Scores[tokenId];
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestIdx = i;
                            bestMerge = candidate;
                        }
                    }
                }

                if (bestIdx >= 0)
                {
                    // Merge the best pair
                    symbols[bestIdx] = bestMerge;
                    symbols.RemoveAt(bestIdx + 1);
                    merged = true;
                }
            }

            // Convert symbols to token IDs
            int pos = 0;
            outputTokens[pos++] = _bosToken;

            for (int i = 0; i < symbols.Count; i++)
            {
                int tokenId = _vocab.GetTokenId(symbols[i]);
                if (tokenId >= 0)
                {
                    outputTokens[pos++] = tokenId;
                }
                else
                {
                    // Unknown token - try byte fallback
                    byte[] tokenBytes = Encoding.UTF8.GetBytes(symbols[i]);
                    for (int b = 0; b < tokenBytes.Length; b++)
                    {
                        if (_byteTokens[tokenBytes[b]] >= 0)
                        {
                            outputTokens[pos++] = _byteTokens[tokenBytes[b]];
                        }
                        else
                        {
                            // Last resort: unknown token (token 0 is usually <unk>)
                            outputTokens[pos++] = 0;
                        }
                    }
                }
            }

            return pos;
        }

        /// <summary>
        /// Decode a single token ID to its string representation.
        /// Handles SentencePiece ▁ to space conversion and byte tokens.
        /// </summary>
        public string Decode(int tokenId)
        {
            if (tokenId < 0 || tokenId >= _vocab.Size)
            {
                return "";
            }

            string token = _vocab.Tokens[tokenId];

            // Handle byte tokens: "<0xHH>" -> actual byte
            if (token.Length == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>')
            {
                if (TryParseHexByte(token, 3, out byte byteVal))
                {
                    // Return the byte as a character
                    Span<byte> bytes = stackalloc byte[1];
                    bytes[0] = byteVal;
                    return Encoding.UTF8.GetString(bytes);
                }
            }

            // Replace SentencePiece ▁ with space
            return token.Replace("\u2581", " ");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryParseHexByte(string s, int offset, out byte value)
        {
            int hi = HexCharToInt(s[offset]);
            int lo = HexCharToInt(s[offset + 1]);
            if (hi >= 0 && lo >= 0)
            {
                value = (byte)((hi << 4) | lo);
                return true;
            }
            value = 0;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HexCharToInt(char c)
        {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return -1;
        }
    }
}
