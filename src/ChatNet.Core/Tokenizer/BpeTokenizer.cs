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
    /// Handles special/control tokens via pre-matching before BPE merge.
    /// </summary>
    public sealed class BpeTokenizer : ITokenizer
    {
        private readonly TokenVocab _vocab;
        private readonly int _bosToken;
        private readonly int _eosToken;

        // Byte fallback tokens: maps byte value -> token ID for "<0xHH>" tokens
        private readonly int[] _byteTokens;

        // Special tokens (control/added) sorted by length descending for greedy matching
        private readonly string[] _specialTokenTexts;
        private readonly int[] _specialTokenIds;
        private readonly int _specialTokenCount;

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

            // Collect special tokens and byte tokens
            var specialTexts = new List<string>();
            var specialIds = new List<int>();

            for (int i = 0; i < tokens.Length; i++)
            {
                string tok = tokens[i];

                // Match "<0xHH>" byte fallback pattern
                if (tok.Length == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>')
                {
                    if (TryParseHexByte(tok, 3, out byte byteVal))
                    {
                        _byteTokens[byteVal] = i;
                    }
                    continue;
                }

                // Identify control/added tokens from token_type array
                // Type 3 = CONTROL, Type 4 = USER_DEFINED
                if (tokenTypes != null && i < tokenTypes.Length)
                {
                    int ttype = tokenTypes[i];
                    if ((ttype == 3 || ttype == 4) && tok.Length > 0)
                    {
                        // Skip BOS (<s>) since it's added explicitly; include all others
                        // including </s> which appears in chat templates
                        if (i != _bosToken)
                        {
                            specialTexts.Add(tok);
                            specialIds.Add(i);
                        }
                    }
                }
            }

            // Sort by length descending for greedy longest-match
            // Simple insertion sort (small list)
            for (int i = 1; i < specialTexts.Count; i++)
            {
                string tmpText = specialTexts[i];
                int tmpId = specialIds[i];
                int j = i - 1;
                while (j >= 0 && specialTexts[j].Length < tmpText.Length)
                {
                    specialTexts[j + 1] = specialTexts[j];
                    specialIds[j + 1] = specialIds[j];
                    j--;
                }
                specialTexts[j + 1] = tmpText;
                specialIds[j + 1] = tmpId;
            }

            _specialTokenTexts = specialTexts.ToArray();
            _specialTokenIds = specialIds.ToArray();
            _specialTokenCount = _specialTokenTexts.Length;
        }

        /// <summary>
        /// Encode text to token IDs using SentencePiece-style BPE.
        /// Pre-matches special/control tokens, then applies BPE to remaining text segments.
        /// Prepends BOS token. Returns number of tokens written.
        /// </summary>
        public int Encode(ReadOnlySpan<char> text, Span<int> outputTokens)
        {
            if (text.Length == 0)
            {
                outputTokens[0] = _bosToken;
                return 1;
            }

            int pos = 0;
            outputTokens[pos++] = _bosToken;

            string fullText = text.ToString();

            if (_specialTokenCount == 0)
            {
                // No special tokens to match; encode entire text with BPE
                pos = EncodeBpeSegment(fullText, outputTokens, pos);
                return pos;
            }

            // Split text at special token boundaries and encode each segment
            int textPos = 0;
            while (textPos < fullText.Length)
            {
                // Try to match a special token at this position (longest match first)
                int matchedId = -1;
                int matchedLen = 0;

                for (int s = 0; s < _specialTokenCount; s++)
                {
                    string specText = _specialTokenTexts[s];
                    if (textPos + specText.Length > fullText.Length)
                        continue;

                    bool match = true;
                    for (int c = 0; c < specText.Length; c++)
                    {
                        if (fullText[textPos + c] != specText[c])
                        {
                            match = false;
                            break;
                        }
                    }

                    if (match)
                    {
                        matchedId = _specialTokenIds[s];
                        matchedLen = specText.Length;
                        break; // Already sorted longest-first, so first match is best
                    }
                }

                if (matchedId >= 0)
                {
                    // Emit the special token directly
                    outputTokens[pos++] = matchedId;
                    textPos += matchedLen;
                }
                else
                {
                    // Find the next special token position (or end of text)
                    int nextSpecialPos = fullText.Length;
                    for (int s = 0; s < _specialTokenCount; s++)
                    {
                        string specText = _specialTokenTexts[s];
                        int idx = fullText.IndexOf(specText, textPos, StringComparison.Ordinal);
                        if (idx >= 0 && idx < nextSpecialPos)
                        {
                            nextSpecialPos = idx;
                        }
                    }

                    // BPE encode the text segment between textPos and nextSpecialPos
                    string segment = fullText.Substring(textPos, nextSpecialPos - textPos);
                    if (segment.Length > 0)
                    {
                        pos = EncodeBpeSegment(segment, outputTokens, pos);
                    }
                    textPos = nextSpecialPos;
                }
            }

            return pos;
        }

        /// <summary>
        /// BPE encode a text segment (no special tokens).
        /// Applies SentencePiece ▁ prefix and space replacement, then iterative BPE merging.
        /// </summary>
        private int EncodeBpeSegment(string text, Span<int> outputTokens, int pos)
        {
            if (text.Length == 0) return pos;

            // SentencePiece: prepend ▁, replace spaces with ▁
            string processed = "\u2581" + text.Replace(" ", "\u2581");

            // Build initial symbol list from individual characters
            var symbols = new List<string>(processed.Length);
            int charIdx = 0;
            while (charIdx < processed.Length)
            {
                string ch = processed.Substring(charIdx, 1);
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
                        string byteTok = "<0x" + charBytes[b].ToString("X2") + ">";
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
                    symbols[bestIdx] = bestMerge;
                    symbols.RemoveAt(bestIdx + 1);
                    merged = true;
                }
            }

            // Convert symbols to token IDs
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
                            outputTokens[pos++] = 0; // <unk>
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

            // Skip control/special tokens in decoded output
            if (_vocab.TokenTypes != null && tokenId < _vocab.TokenTypes.Length)
            {
                int ttype = _vocab.TokenTypes[tokenId];
                if (ttype == 3) return ""; // CONTROL tokens like <s>, </s>
            }

            // Handle byte tokens: "<0xHH>" -> actual byte
            if (token.Length == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>')
            {
                if (TryParseHexByte(token, 3, out byte byteVal))
                {
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
