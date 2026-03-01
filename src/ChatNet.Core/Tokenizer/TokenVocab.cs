using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

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

        // Merge lookup: avoids allocating a concatenated string for every BPE candidate.
        // Key = hash of (left + right), Value = list of (leftLen, tokenId) pairs to disambiguate collisions.
        private readonly Dictionary<int, MergeBucket> _mergeMap;

        // ID-based merge lookup: key = packed (leftId << 32 | rightId), value = merged token ID.
        // Used by the hot BPE merge loop to avoid string operations entirely.
        private readonly Dictionary<long, int> _mergeIdMap;

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

            // Build merge map: for each token that could be the result of merging two
            // existing tokens, record it keyed by the hash of the concatenation.
            _mergeMap = new Dictionary<int, MergeBucket>(tokens.Length);
            _mergeIdMap = new Dictionary<long, int>(tokens.Length);
            for (int i = 0; i < tokens.Length; i++)
            {
                string tok = tokens[i];
                if (tok.Length < 2) continue;

                // Try every split point: left = tok[..s], right = tok[s..]
                for (int s = 1; s < tok.Length; s++)
                {
                    string left = tok.Substring(0, s);
                    string right = tok.Substring(s);

                    if (_tokenToId.TryGetValue(left, out int leftId) && _tokenToId.TryGetValue(right, out int rightId))
                    {
                        int hash = ConcatHash(left, right);
                        if (!_mergeMap.TryGetValue(hash, out var bucket))
                        {
                            bucket = new MergeBucket();
                            _mergeMap[hash] = bucket;
                        }
                        bucket.Add(left, right, i);

                        // ID-based merge map: no string ops in hot merge loop
                        long idKey = ((long)leftId << 32) | (uint)rightId;
                        if (!_mergeIdMap.ContainsKey(idKey))
                            _mergeIdMap[idKey] = i;
                    }
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

        /// <summary>
        /// Look up the token ID for the concatenation of left + right without
        /// allocating the concatenated string. Returns -1 if not found.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetMergeTokenId(string left, string right)
        {
            int hash = ConcatHash(left, right);
            if (_mergeMap.TryGetValue(hash, out var bucket))
            {
                return bucket.Lookup(left, right);
            }
            return -1;
        }

        /// <summary>
        /// Look up the merged token ID by left and right token IDs, without any string
        /// operations. Returns -1 if no such merge exists in the vocabulary.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetMergeTokenIdById(int leftId, int rightId)
        {
            // Pack two 32-bit IDs into one 64-bit key.
            // (uint) cast is intentional: it reinterprets the sign bit as a value bit,
            // keeping the mapping bijective (same cast is used when inserting).
            long key = ((long)leftId << 32) | (uint)rightId;
            return _mergeIdMap.TryGetValue(key, out int id) ? id : -1;
        }

        /// <summary>Check if a token string exists in vocab.</summary>
        public bool Contains(string token)
        {
            return _tokenToId.ContainsKey(token);
        }

        /// <summary>Compute a hash for the concatenation of two strings without allocating.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ConcatHash(string a, string b)
        {
            // FNV-1a inspired hash over both strings sequentially
            uint h = 2166136261u;
            for (int i = 0; i < a.Length; i++)
            {
                h = (h ^ a[i]) * 16777619u;
            }
            for (int i = 0; i < b.Length; i++)
            {
                h = (h ^ b[i]) * 16777619u;
            }
            return (int)h;
        }

        /// <summary>
        /// Small collision bucket for merge lookups. Most buckets have 1-2 entries.
        /// </summary>
        private sealed class MergeBucket
        {
            private string[] _lefts = new string[2];
            private string[] _rights = new string[2];
            private int[] _ids = new int[2];
            private int _count;

            public void Add(string left, string right, int tokenId)
            {
                // Dedup: same merge result can be registered from the same token at one split point
                for (int i = 0; i < _count; i++)
                {
                    if (_lefts[i] == left && _rights[i] == right)
                        return;
                }

                if (_count == _lefts.Length)
                {
                    int newLen = _lefts.Length * 2;
                    Array.Resize(ref _lefts, newLen);
                    Array.Resize(ref _rights, newLen);
                    Array.Resize(ref _ids, newLen);
                }
                _lefts[_count] = left;
                _rights[_count] = right;
                _ids[_count] = tokenId;
                _count++;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public int Lookup(string left, string right)
            {
                for (int i = 0; i < _count; i++)
                {
                    if (_lefts[i].Length == left.Length && _rights[i].Length == right.Length &&
                        _lefts[i] == left && _rights[i] == right)
                    {
                        return _ids[i];
                    }
                }
                return -1;
            }
        }
    }
}
