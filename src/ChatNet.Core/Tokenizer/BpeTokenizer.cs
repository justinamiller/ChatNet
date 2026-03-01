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
    /// Handles special/control tokens via trie-based pre-matching before BPE merge.
    /// Operates on int token IDs with pooled buffers in the hot path.
    /// </summary>
    public sealed class BpeTokenizer : ITokenizer
    {
        private readonly TokenVocab _vocab;
        private readonly int _bosToken;
        private readonly int _eosToken;

        // Byte fallback tokens: maps byte value (0-255) -> token ID for "<0xHH>" tokens
        private readonly int[] _byteTokens;

        // Pre-computed decode cache: tokenId -> decoded string (refactor item 4)
        private readonly string[] _decodedCache;

        // Trie for single-pass special token matching (refactor item 2)
        private readonly TrieNode? _specialTokenTrie;
        private readonly bool _hasSpecialTokens;

        // Char -> token ID caches for fast initial symbol building (refactor item 1)
        private readonly int[] _asciiTokenIds;                     // ASCII 0-127 -> token ID (-1 if absent)
        private readonly int _spSpaceTokenId;                      // SentencePiece ▁ (U+2581) token ID
        private readonly Dictionary<char, int> _unicodeCharTokenIds; // non-ASCII single-char tokens

        // Static cache for single-char strings (used only during construction)
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

            // Build char -> tokenId caches (refactor item 1)
            _asciiTokenIds = new int[128];
            for (int i = 0; i < 128; i++)
                _asciiTokenIds[i] = _vocab.GetTokenId(s_charStrings[i]);
            _spSpaceTokenId = _vocab.GetTokenId("\u2581");

            _unicodeCharTokenIds = new Dictionary<char, int>();

            // Scan vocab for byte tokens, special tokens, and unicode char tokens
            var trieRoot = new TrieNode();
            bool hasSpecial = false;

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

                // Cache single-char unicode tokens (non-ASCII, non-▁)
                if (tok.Length == 1)
                {
                    char c = tok[0];
                    if (c >= 128 && c != '\u2581' && !_unicodeCharTokenIds.ContainsKey(c))
                        _unicodeCharTokenIds[c] = i;
                }

                // Identify control/added tokens -> insert into trie (refactor item 2)
                // Type 3 = CONTROL, Type 4 = USER_DEFINED
                if (tokenTypes != null && i < tokenTypes.Length)
                {
                    int ttype = tokenTypes[i];
                    if ((ttype == 3 || ttype == 4) && tok.Length > 0)
                    {
                        // Skip BOS (<s>) since it's added explicitly
                        if (i != _bosToken)
                        {
                            TrieNode node = trieRoot;
                            for (int c = 0; c < tok.Length; c++)
                                node = node.GetOrAddChild(tok[c]);
                            node.TokenId = i;
                            hasSpecial = true;
                        }
                    }
                }
            }

            _hasSpecialTokens = hasSpecial;
            _specialTokenTrie = hasSpecial ? trieRoot : null;

            // Build decode cache: pre-compute decoded string for every token (refactor item 4)
            _decodedCache = new string[_vocab.Size];
            for (int i = 0; i < _vocab.Size; i++)
                _decodedCache[i] = ComputeDecoded(i);
        }

        /// <summary>
        /// Pre-compute the decoded string for a token ID (used once in constructor).
        /// </summary>
        private string ComputeDecoded(int tokenId)
        {
            string token = _vocab.Tokens[tokenId];

            // Skip control tokens in decoded output
            if (_vocab.TokenTypes != null && tokenId < _vocab.TokenTypes.Length)
            {
                if (_vocab.TokenTypes[tokenId] == 3) return "";
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

        /// <summary>
        /// Decode a single token ID to its string representation.
        /// O(1) cache lookup with no Replace or allocation (refactor item 4).
        /// </summary>
        public string Decode(int tokenId)
        {
            if ((uint)tokenId >= (uint)_decodedCache.Length)
                return "";
            return _decodedCache[tokenId];
        }

        /// <summary>
        /// Encode text to token IDs using SentencePiece-style BPE.
        /// Uses trie-based single-pass scanning for special/control tokens (refactor item 2).
        /// Prepends BOS token. Returns number of tokens written.
        /// </summary>
        public int Encode(ReadOnlySpan<char> text, Span<int> outputTokens)
        {
            // Output capacity check (refactor item 5)
            if (outputTokens.Length == 0) return 0;

            if (text.Length == 0)
            {
                outputTokens[0] = _bosToken;
                return 1;
            }

            int pos = 0;
            outputTokens[pos++] = _bosToken;

            if (!_hasSpecialTokens)
            {
                // No special tokens to match; encode entire text with BPE
                pos = EncodeBpeSegment(text, outputTokens, pos);
                return pos;
            }

            // Single-pass trie-based special token scanning (refactor item 2)
            // Walk text left-to-right, match longest special token at each position
            int segmentStart = 0;
            int textPos = 0;

            while (textPos < text.Length)
            {
                if (pos >= outputTokens.Length) return pos;

                // Try longest-prefix match at this position using trie
                TrieNode? node = _specialTokenTrie;
                int matchedId = -1;
                int matchedLen = 0;

                for (int i = textPos; i < text.Length; i++)
                {
                    node = node!.GetChild(text[i]);
                    if (node == null) break;
                    if (node.TokenId >= 0)
                    {
                        matchedId = node.TokenId;
                        matchedLen = i - textPos + 1;
                        // Continue to find longest match
                    }
                }

                if (matchedId >= 0)
                {
                    // BPE-encode any text before this special token
                    if (textPos > segmentStart && pos < outputTokens.Length)
                    {
                        pos = EncodeBpeSegment(
                            text.Slice(segmentStart, textPos - segmentStart),
                            outputTokens, pos);
                    }

                    // Emit the special token directly
                    if (pos < outputTokens.Length)
                        outputTokens[pos++] = matchedId;

                    textPos += matchedLen;
                    segmentStart = textPos;
                }
                else
                {
                    textPos++;
                }
            }

            // Encode remaining text after last special token
            if (segmentStart < text.Length && pos < outputTokens.Length)
            {
                pos = EncodeBpeSegment(
                    text.Slice(segmentStart, text.Length - segmentStart),
                    outputTokens, pos);
            }

            return pos;
        }

        /// <summary>
        /// BPE encode a text segment (no special tokens).
        /// Uses int token IDs with linked-list structure and pooled buffers (refactor items 1, 3, 6).
        /// Applies SentencePiece ▁ prefix and space replacement, then iterative BPE merging.
        /// </summary>
        private int EncodeBpeSegment(ReadOnlySpan<char> text, Span<int> outputTokens, int pos)
        {
            if (text.Length == 0 || pos >= outputTokens.Length) return pos;

            // Max symbols: ▁ prefix (up to 3 byte tokens) + text.Length * 4 (worst case byte fallback)
            int maxSymbols = 4 + text.Length * 4;

            // Rent pooled buffers for linked-list symbol representation (refactor item 3)
            int[] symIds = ArrayPool<int>.Shared.Rent(maxSymbols);
            int[] symPrev = ArrayPool<int>.Shared.Rent(maxSymbols);
            int[] symNext = ArrayPool<int>.Shared.Rent(maxSymbols);

            // Build processed char buffer: ▁ prefix + text with spaces replaced by ▁
            // Encode entire buffer to UTF-8 once for byte fallback (refactor item 6)
            int processedCharLen = 1 + text.Length;
            char[] procChars = ArrayPool<char>.Shared.Rent(processedCharLen);
            procChars[0] = '\u2581';
            for (int i = 0; i < text.Length; i++)
                procChars[i + 1] = text[i] == ' ' ? '\u2581' : text[i];

            int utf8Len = Encoding.UTF8.GetByteCount(procChars, 0, processedCharLen);
            byte[] utf8Buf = ArrayPool<byte>.Shared.Rent(utf8Len);
            Encoding.UTF8.GetBytes(procChars, 0, processedCharLen, utf8Buf, 0);

            try
            {
                int symCount = 0;
                int byteOff = 0;

                // Build initial symbol list from chars as int token IDs (refactor item 1)
                for (int i = 0; i < processedCharLen; i++)
                {
                    char c = procChars[i];

                    // Compute UTF-8 byte length for this char (for byte offset tracking)
                    int charByteLen;
                    if (c < 0x80)
                        charByteLen = 1;
                    else if (c < 0x800)
                        charByteLen = 2;
                    else if (char.IsHighSurrogate(c) && i + 1 < processedCharLen &&
                             char.IsLowSurrogate(procChars[i + 1]))
                        charByteLen = 4;
                    else
                        charByteLen = 3;

                    // Look up char's token ID from pre-built caches (no string allocation)
                    int tokenId;
                    if (c == '\u2581')
                        tokenId = _spSpaceTokenId;
                    else if (c < 128)
                        tokenId = _asciiTokenIds[c];
                    else if (!_unicodeCharTokenIds.TryGetValue(c, out tokenId))
                        tokenId = -1;

                    if (tokenId >= 0)
                    {
                        // Append to linked list
                        symIds[symCount] = tokenId;
                        symPrev[symCount] = symCount > 0 ? symCount - 1 : -1;
                        symNext[symCount] = -1;
                        if (symCount > 0) symNext[symCount - 1] = symCount;
                        symCount++;
                    }
                    else
                    {
                        // Byte fallback: emit byte tokens from pre-encoded UTF-8 buffer (refactor item 6)
                        for (int b = 0; b < charByteLen; b++)
                        {
                            int bt = _byteTokens[utf8Buf[byteOff + b]];
                            symIds[symCount] = bt >= 0 ? bt : 0; // fallback to <unk>
                            symPrev[symCount] = symCount > 0 ? symCount - 1 : -1;
                            symNext[symCount] = -1;
                            if (symCount > 0) symNext[symCount - 1] = symCount;
                            symCount++;
                        }
                    }

                    byteOff += charByteLen;
                    if (charByteLen == 4) i++; // skip low surrogate of surrogate pair
                }

                if (symCount == 0) return pos;

                // BPE merge loop: repeatedly find and apply the highest-scoring merge.
                // Uses int-based merge lookup (refactor item 1) with linked-list traversal.
                int liveCount = symCount;
                bool merged = true;
                while (merged && liveCount > 1)
                {
                    merged = false;
                    float bestScore = float.NegativeInfinity;
                    int bestIdx = -1;
                    int bestMergeId = -1;

                    // Walk linked list to find best adjacent pair
                    int cur = 0;
                    while (true)
                    {
                        int nxt = symNext[cur];
                        if (nxt == -1) break;

                        int mergeId = _vocab.GetMergeTokenIdById(symIds[cur], symIds[nxt]);
                        if (mergeId >= 0)
                        {
                            float score = _vocab.Scores[mergeId];
                            if (score > bestScore)
                            {
                                bestScore = score;
                                bestIdx = cur;
                                bestMergeId = mergeId;
                            }
                        }
                        cur = nxt;
                    }

                    if (bestIdx >= 0)
                    {
                        // Apply merge: replace left symbol with merged token, remove right from list
                        int removeIdx = symNext[bestIdx];
                        symIds[bestIdx] = bestMergeId;

                        int afterRemove = symNext[removeIdx];
                        symNext[bestIdx] = afterRemove;
                        if (afterRemove != -1)
                            symPrev[afterRemove] = bestIdx;

                        liveCount--;
                        merged = true;
                    }
                }

                // Output: walk linked list and write token IDs (refactor item 5: capacity checks)
                int outCur = 0;
                while (outCur != -1 && pos < outputTokens.Length)
                {
                    outputTokens[pos++] = symIds[outCur];
                    outCur = symNext[outCur];
                }

                return pos;
            }
            finally
            {
                ArrayPool<int>.Shared.Return(symIds);
                ArrayPool<int>.Shared.Return(symPrev);
                ArrayPool<int>.Shared.Return(symNext);
                ArrayPool<char>.Shared.Return(procChars);
                ArrayPool<byte>.Shared.Return(utf8Buf);
            }
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

        /// <summary>
        /// Trie node for efficient special token prefix matching.
        /// Uses Dictionary for children since special tokens are few.
        /// </summary>
        private sealed class TrieNode
        {
            public int TokenId = -1;
            private Dictionary<char, TrieNode>? _children;

            public TrieNode? GetChild(char c)
            {
                if (_children == null) return null;
                _children.TryGetValue(c, out var node);
                return node;
            }

            public TrieNode GetOrAddChild(char c)
            {
                _children ??= new Dictionary<char, TrieNode>();
                if (!_children.TryGetValue(c, out var node))
                {
                    node = new TrieNode();
                    _children[c] = node;
                }
                return node;
            }
        }
    }
}
