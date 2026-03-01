using System;
using System.Buffers;
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

        // Per-token decoded string cache. Built once in ctor; Decode is O(1).
        // Handles ▁→space replacement, byte-token decoding, and control-token suppression upfront.
        private readonly string[] _decodeCache;

        // Cached single-char strings to avoid Substring(i, 1) allocations in BPE init loop.
        // Covers ASCII + SentencePiece ▁ (U+2581). Indexed by char value for chars < 128,
        // with ▁ at index 128.
        private static readonly string[] s_charStrings = BuildCharStringCache();

        private static string[] BuildCharStringCache()
        {
            // 129 entries: 0..127 for ASCII, 128 for ▁
            var cache = new string[129];
            for (int i = 0; i < 128; i++)
                cache[i] = ((char)i).ToString();
            cache[128] = "\u2581";
            return cache;
        }

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

            // Build decode cache: precompute the decoded string for every token ID once.
            // Decode() becomes a single array lookup with no per-call allocations.
            _decodeCache = new string[tokens.Length];
            Span<byte> byteDecodeBuffer = stackalloc byte[1];
            for (int i = 0; i < tokens.Length; i++)
            {
                string tok = tokens[i];

                // Control tokens (type 3) produce no output text
                if (tokenTypes != null && i < tokenTypes.Length && tokenTypes[i] == 3)
                {
                    _decodeCache[i] = "";
                    continue;
                }

                // Byte token "<0xHH>" → the actual UTF-8 byte as a string
                if (tok.Length == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>')
                {
                    if (TryParseHexByte(tok, 3, out byte byteVal))
                    {
                        byteDecodeBuffer[0] = byteVal;
                        _decodeCache[i] = Encoding.UTF8.GetString(byteDecodeBuffer);
                        continue;
                    }
                }

                // Replace SentencePiece ▁ with space (only allocate a new string when needed)
                _decodeCache[i] = tok.IndexOf('\u2581') >= 0 ? tok.Replace("\u2581", " ") : tok;
            }
        }

        /// <summary>
        /// Encode text to token IDs using SentencePiece-style BPE.
        /// Pre-matches special/control tokens, then applies BPE to remaining text segments.
        /// Prepends BOS token. Returns number of tokens written.
        /// </summary>
        public int Encode(ReadOnlySpan<char> text, Span<int> outputTokens)
        {
            if (outputTokens.Length == 0) return 0;

            if (text.Length == 0)
            {
                outputTokens[0] = _bosToken;
                return 1;
            }

            int pos = 0;
            outputTokens[pos++] = _bosToken;

            if (_specialTokenCount == 0)
            {
                // No special tokens: encode entire text with BPE (no string allocation)
                pos = EncodeBpeSegment(text, outputTokens, pos);
                return pos;
            }

            // Split text at special-token boundaries and BPE-encode each plain segment
            int textPos = 0;
            int textLen = text.Length;

            while (textPos < textLen)
            {
                // Try to match a special token at current position (longest-first)
                int matchedId = -1;
                int matchedLen = 0;

                for (int s = 0; s < _specialTokenCount; s++)
                {
                    string specText = _specialTokenTexts[s];
                    int specLen = specText.Length;
                    if (textPos + specLen > textLen) continue;

                    bool match = true;
                    for (int c = 0; c < specLen; c++)
                    {
                        if (text[textPos + c] != specText[c])
                        {
                            match = false;
                            break;
                        }
                    }

                    if (match)
                    {
                        matchedId = _specialTokenIds[s];
                        matchedLen = specLen;
                        break; // Already sorted longest-first
                    }
                }

                if (matchedId >= 0)
                {
                    if (pos < outputTokens.Length)
                        outputTokens[pos++] = matchedId;
                    textPos += matchedLen;
                }
                else
                {
                    // Find the start of the next special token (or end of text)
                    int nextSpecialPos = textLen;
                    for (int s = 0; s < _specialTokenCount; s++)
                    {
                        int idx = text.Slice(textPos).IndexOf(_specialTokenTexts[s].AsSpan());
                        if (idx >= 0)
                        {
                            int absIdx = textPos + idx;
                            if (absIdx < nextSpecialPos)
                                nextSpecialPos = absIdx;
                        }
                    }

                    // BPE-encode the plain segment between textPos and nextSpecialPos
                    if (nextSpecialPos > textPos)
                        pos = EncodeBpeSegment(text.Slice(textPos, nextSpecialPos - textPos), outputTokens, pos);

                    textPos = nextSpecialPos;
                }
            }

            return pos;
        }

        /// <summary>
        /// BPE encode a text segment (no special tokens).
        /// Applies SentencePiece ▁ prefix and space replacement, then iterative BPE merging.
        /// Uses ArrayPool-rented int arrays and a doubly-linked list for O(1) merge removal.
        /// No heap allocations except for rare non-ASCII character token lookups.
        /// </summary>
        private int EncodeBpeSegment(ReadOnlySpan<char> text, Span<int> outputTokens, int pos)
        {
            if (text.Length == 0) return pos;

            // Upper bound: leading ▁ + each input char can expand to at most 4 UTF-8 byte tokens.
            // Guard against int overflow for pathologically large inputs.
            int maxSymbols = text.Length <= (int.MaxValue / 4) - 1
                ? (text.Length + 1) * 4
                : int.MaxValue;

            int[] symIds  = ArrayPool<int>.Shared.Rent(maxSymbols);
            int[] symNext = ArrayPool<int>.Shared.Rent(maxSymbols);
            int[] symPrev = ArrayPool<int>.Shared.Rent(maxSymbols);

            try
            {
                int symCount = 0;
                Span<byte> utf8Buf = stackalloc byte[4];
                Span<char> charBuf = stackalloc char[1];

                // Process: leading ▁, then each char of text (spaces become ▁)
                for (int ci = -1; ci < text.Length; ci++)
                {
                    char c = ci < 0 ? '\u2581' : (text[ci] == ' ' ? '\u2581' : text[ci]);

                    int tokenId;
                    if (c < 128)
                        tokenId = _vocab.GetTokenId(s_charStrings[(int)c]);
                    else if (c == '\u2581')
                        tokenId = _vocab.GetTokenId(s_charStrings[128]);
                    else
                        tokenId = _vocab.GetTokenId(c.ToString()); // rare non-ASCII non-▁ char

                    if (tokenId >= 0)
                    {
                        symIds[symCount]  = tokenId;
                        symPrev[symCount] = symCount - 1;
                        symNext[symCount] = symCount + 1;
                        symCount++;
                    }
                    else
                    {
                        // UTF-8 byte fallback – encode to bytes without heap allocation
                        charBuf[0] = c;
                        int byteLen = Encoding.UTF8.GetBytes(charBuf, utf8Buf);
                        for (int b = 0; b < byteLen; b++)
                        {
                            int byteId = _byteTokens[utf8Buf[b]];
                            symIds[symCount]  = byteId >= 0 ? byteId : 0;
                            symPrev[symCount] = symCount - 1;
                            symNext[symCount] = symCount + 1;
                            symCount++;
                        }
                    }
                }

                if (symCount == 0) return pos;

                // Fix boundary nodes
                symPrev[0] = -1;
                symNext[symCount - 1] = -1;

                // BPE merge loop: find best adjacent pair by score, merge, repeat.
                // Linked list gives O(1) removal; ID-based merge lookup avoids string ops.
                // All symNext values are either a valid index in [0, symCount) or -1 (sentinel),
                // so the loop index stays within the allocated range at all times.
                while (symCount > 1)
                {
                    float bestScore   = float.NegativeInfinity;
                    int   bestIdx     = -1;
                    int   bestMergeId = -1;

                    for (int i = 0; i >= 0 && symNext[i] != -1;)
                    {
                        int j = symNext[i];
                        int mergeId = _vocab.GetMergeTokenIdById(symIds[i], symIds[j]);
                        if (mergeId >= 0)
                        {
                            float score = _vocab.Scores[mergeId];
                            if (score > bestScore)
                            {
                                bestScore   = score;
                                bestIdx     = i;
                                bestMergeId = mergeId;
                            }
                        }
                        i = j;
                    }

                    if (bestIdx < 0) break; // no more merges possible

                    // Merge: update left slot, unlink right slot in O(1)
                    symIds[bestIdx] = bestMergeId;
                    int killed = symNext[bestIdx];
                    symNext[bestIdx] = symNext[killed];
                    if (symNext[killed] != -1)
                        symPrev[symNext[killed]] = bestIdx;
                    symCount--;
                }

                // Emit token IDs in linked-list order
                for (int i = 0; i != -1 && pos < outputTokens.Length; i = symNext[i])
                    outputTokens[pos++] = symIds[i];
            }
            finally
            {
                ArrayPool<int>.Shared.Return(symIds);
                ArrayPool<int>.Shared.Return(symNext);
                ArrayPool<int>.Shared.Return(symPrev);
            }

            return pos;
        }

        /// <summary>
        /// Decode a single token ID to its string representation.
        /// Result is returned from a pre-built cache; O(1), no per-call allocations.
        /// </summary>
        public string Decode(int tokenId)
        {
            if ((uint)tokenId >= (uint)_decodeCache.Length)
                return "";
            return _decodeCache[tokenId];
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

