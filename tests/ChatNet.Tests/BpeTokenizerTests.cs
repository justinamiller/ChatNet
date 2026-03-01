using System;
using ChatNet.Core.Gguf;
using ChatNet.Core.Tokenizer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ChatNet.Tests
{
    [TestClass]
    public class BpeTokenizerTests
    {
        // ── helpers ────────────────────────────────────────────────────────────

        /// <summary>
        /// Build a minimal GgufMetadata that the BpeTokenizer can consume.
        /// Vocabulary layout:
        ///   0  = &lt;unk&gt;  (normal)
        ///   1  = &lt;s&gt;   (control / BOS)
        ///   2  = &lt;/s&gt;  (control / EOS)
        ///   3  = "▁"
        ///   4  = "h"
        ///   5  = "e"
        ///   6  = "l"
        ///   7  = "o"
        ///   8  = "▁h"    ← merge of 3+4
        ///   9  = "▁he"   ← merge of 8+5
        ///   10 = "ll"    ← merge of 6+6
        ///   11 = "llo"   ← merge of 10+7
        ///   12 = "&lt;0x20&gt;"  byte-fallback for 0x20 (space / ASCII 32)
        ///   13 = "[INST]"  user-defined special token
        /// </summary>
        private static BpeTokenizer BuildTokenizer()
        {
            string[] tokens =
            {
                "<unk>",    // 0
                "<s>",      // 1  BOS
                "</s>",     // 2  EOS
                "\u2581",   // 3  ▁
                "h",        // 4
                "e",        // 5
                "l",        // 6
                "o",        // 7
                "\u2581h",  // 8   merge(▁, h)
                "\u2581he", // 9   merge(▁h, e)
                "ll",       // 10  merge(l, l)
                "llo",      // 11  merge(ll, o)
                "<0x20>",   // 12  byte-fallback for 0x20
                "[INST]",   // 13  user-defined special token
            };

            // Scores: higher = preferred merge.  BPE should prefer longer merges.
            float[] scores =
            {
                0f,  // 0  <unk>
                0f,  // 1  <s>
                0f,  // 2  </s>
                -1f, // 3  ▁
                -2f, // 4  h
                -3f, // 5  e
                -4f, // 6  l
                -5f, // 7  o
                5f,  // 8  ▁h
                10f, // 9  ▁he
                8f,  // 10 ll
                12f, // 11 llo
                0f,  // 12 <0x20>
                0f,  // 13 [INST]
            };

            int[] tokenTypes =
            {
                2, // 0  UNKNOWN
                3, // 1  CONTROL  (BOS)
                3, // 2  CONTROL  (EOS)
                1, // 3  NORMAL
                1, // 4  NORMAL
                1, // 5  NORMAL
                1, // 6  NORMAL
                1, // 7  NORMAL
                1, // 8  NORMAL
                1, // 9  NORMAL
                1, // 10 NORMAL
                1, // 11 NORMAL
                1, // 12 NORMAL  (byte token – not type 3)
                4, // 13 USER_DEFINED (special token)
            };

            var meta = new GgufMetadata();
            meta.Set("tokenizer.ggml.tokens",       tokens);
            meta.Set("tokenizer.ggml.scores",        scores);
            meta.Set("tokenizer.ggml.token_type",    tokenTypes);
            meta.Set("tokenizer.ggml.bos_token_id",  1);
            meta.Set("tokenizer.ggml.eos_token_id",  2);

            return new BpeTokenizer(meta);
        }

        // ── Encode tests ───────────────────────────────────────────────────────

        [TestMethod]
        public void Encode_EmptyText_ReturnsBosOnly()
        {
            var tok = BuildTokenizer();
            Span<int> buf = stackalloc int[16];

            int n = tok.Encode(ReadOnlySpan<char>.Empty, buf);

            Assert.AreEqual(1, n);
            Assert.AreEqual(tok.BosToken, buf[0]);
        }

        [TestMethod]
        public void Encode_SimpleWord_AppliesBpeAndPrependsLeadingUnderscore()
        {
            // "hello" → ▁ + h + e + l + l + o
            // BPE with given scores should merge:
            //   1) (ll, o) → "llo"  (score 12, best)
            //   2) (▁, h) → "▁h"   wait — let's check: after "llo" merge we have [▁,h,e,llo]
            //   3) (▁h, e) → "▁he"  (score 10 vs ▁h score 5 – "▁he" wins first? No, we pick highest score across ALL pairs each iteration)
            //
            // Iteration 1 – scan all pairs in [▁,h,e,l,l,o]:
            //   (▁,h)→▁h   score=5
            //   (h,e)→?     not in vocab → -1
            //   (e,l)→?     not in vocab → -1
            //   (l,l)→ll    score=8
            //   (l,o)→?     not in vocab → -1
            //   best = ll (score 8) → [▁,h,e,ll,o]
            //
            // Iteration 2 – scan [▁,h,e,ll,o]:
            //   (▁,h)→▁h  score=5
            //   (h,e)→?    -1
            //   (e,ll)→?   -1
            //   (ll,o)→llo score=12  ← best
            //   → [▁,h,e,llo]
            //
            // Iteration 3 – scan [▁,h,e,llo]:
            //   (▁,h)→▁h  score=5  ← best
            //   (h,e)→?    -1
            //   (e,llo)→?  -1
            //   → [▁h,e,llo]
            //
            // Iteration 4 – scan [▁h,e,llo]:
            //   (▁h,e)→▁he score=10 ← best
            //   (e,llo)→?  -1
            //   → [▁he,llo]
            //
            // Iteration 5 – scan [▁he,llo]:
            //   (▁he,llo)→? not in vocab → -1
            //   no merge → stop
            //
            // Final token IDs: BOS(1), 9(▁he), 11(llo)

            var tok = BuildTokenizer();
            Span<int> buf = stackalloc int[16];

            int n = tok.Encode("hello".AsSpan(), buf);

            Assert.AreEqual(3, n, "Expected BOS + 2 merged tokens");
            Assert.AreEqual(1,  buf[0], "BOS");
            Assert.AreEqual(9,  buf[1], "▁he");
            Assert.AreEqual(11, buf[2], "llo");
        }

        [TestMethod]
        public void Encode_SpecialTokenInText_EmittedDirectly()
        {
            // "[INST]hello" → BOS, [INST](13), BPE("hello")
            var tok = BuildTokenizer();
            Span<int> buf = stackalloc int[32];

            int n = tok.Encode("[INST]hello".AsSpan(), buf);

            Assert.IsTrue(n >= 3);
            Assert.AreEqual(1,  buf[0], "BOS");
            Assert.AreEqual(13, buf[1], "[INST] special token");
            // BPE result for "hello" starts at buf[2]
            Assert.AreEqual(9,  buf[2], "▁he");
            Assert.AreEqual(11, buf[3], "llo");
        }

        [TestMethod]
        public void Encode_OutputBufferFull_DoesNotOverrun()
        {
            var tok = BuildTokenizer();
            // Tiny buffer: only room for BOS
            Span<int> buf = stackalloc int[1];

            int n = tok.Encode("hello".AsSpan(), buf);

            // Should write exactly 1 (BOS) and not throw
            Assert.AreEqual(1, n);
            Assert.AreEqual(1, buf[0]);
        }

        [TestMethod]
        public void Encode_EmptyOutputBuffer_DoesNotThrow()
        {
            var tok = BuildTokenizer();
            Span<int> buf = Span<int>.Empty;

            int n = tok.Encode(ReadOnlySpan<char>.Empty, buf);
            Assert.AreEqual(0, n);
        }

        // ── Decode tests ───────────────────────────────────────────────────────

        [TestMethod]
        public void Decode_ControlToken_ReturnsEmpty()
        {
            var tok = BuildTokenizer();

            Assert.AreEqual("", tok.Decode(1), "BOS (control) should decode to empty");
            Assert.AreEqual("", tok.Decode(2), "EOS (control) should decode to empty");
        }

        [TestMethod]
        public void Decode_NormalToken_ReplacesUnderscore()
        {
            var tok = BuildTokenizer();

            // token 8 = "▁h" → " h"
            Assert.AreEqual(" h", tok.Decode(8));
            // token 9 = "▁he" → " he"
            Assert.AreEqual(" he", tok.Decode(9));
        }

        [TestMethod]
        public void Decode_ByteToken_ReturnsActualChar()
        {
            var tok = BuildTokenizer();

            // token 12 = "<0x20>" → " " (space)
            string decoded = tok.Decode(12);
            Assert.AreEqual(" ", decoded);
        }

        [TestMethod]
        public void Decode_OutOfRangeId_ReturnsEmpty()
        {
            var tok = BuildTokenizer();
            Assert.AreEqual("", tok.Decode(-1));
            Assert.AreEqual("", tok.Decode(9999));
        }

        // ── TokenVocab merge-by-ID tests ───────────────────────────────────────

        [TestMethod]
        public void TokenVocab_GetMergeTokenIdById_ReturnsCorrectId()
        {
            string[] tokens = { "a", "b", "ab" };
            float[]  scores = { 0f, 0f, 1f };
            var vocab = new TokenVocab(tokens, scores, null);

            // a=0, b=1, ab=2
            int merged = vocab.GetMergeTokenIdById(0, 1);
            Assert.AreEqual(2, merged, "merge(a,b)=ab should map to id 2");
        }

        [TestMethod]
        public void TokenVocab_GetMergeTokenIdById_ReturnsNegativeOneForNonexistentPair()
        {
            string[] tokens = { "a", "b", "c" };
            float[]  scores = { 0f, 0f, 0f };
            var vocab = new TokenVocab(tokens, scores, null);

            int merged = vocab.GetMergeTokenIdById(0, 1);
            Assert.AreEqual(-1, merged);
        }
    }
}
