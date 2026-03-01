using System;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q6_K dequantization: super-blocks of 256 values.
    /// Each block = ql[128] + qh[64] + scales[16] + d[2] = 210 bytes.
    ///
    /// Layout per block:
    ///   ql[128]   : lower 4 bits of 6-bit quants (2 values per byte via nibbles)
    ///   qh[64]    : upper 2 bits of 6-bit quants (4 values per byte via bit pairs)
    ///   scales[16]: int8 sub-block scales (one per 16 elements)
    ///   d[2]      : FP16 super-block scale
    ///
    /// Dequantization: value = d * scales[group] * (q6 - 32)
    /// where q6 = (ql_nibble) | (qh_bits &lt;&lt; 4), range 0..63
    /// </summary>
    public static class DequantQ6K
    {
        /// <summary>Block size for Q6_K: 256 elements per super-block.</summary>
        public const int BlockSize = 256;

        /// <summary>Bytes per block: 128 + 64 + 16 + 2 = 210.</summary>
        public const int BytesPerBlock = 210;

        // Byte offsets within a block
        private const int QlOffset = 0;
        private const int QhOffset = 128;
        private const int ScalesOffset = 192;
        private const int DOffset = 208;

        /// <summary>
        /// Dequantize Q6_K data into a float output buffer.
        /// </summary>
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DOffset],
                    quantizedData[srcOffset + DOffset + 1]);

                // Process 256 elements in two halves of 128
                int qlBase = srcOffset + QlOffset;
                int qhBase = srcOffset + QhOffset;
                int scBase = srcOffset + ScalesOffset;

                for (int half = 0; half < 2; half++)
                {
                    int qlOff = qlBase + half * 64;
                    int qhOff = qhBase + half * 32;
                    int scOff = half * 8; // scale index offset

                    for (int l = 0; l < 32; l++)
                    {
                        int scIdx = scOff + l / 16;

                        int q1 = ((quantizedData[qlOff + l] & 0x0F) |
                                  (((quantizedData[qhOff + l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((quantizedData[qlOff + l + 32] & 0x0F) |
                                  (((quantizedData[qhOff + l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((quantizedData[qlOff + l] >> 4) |
                                  (((quantizedData[qhOff + l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((quantizedData[qlOff + l + 32] >> 4) |
                                  (((quantizedData[qhOff + l] >> 6) & 3) << 4)) - 32;

                        int outBase = dstOffset + half * 128;
                        sbyte sc0 = (sbyte)quantizedData[scBase + scIdx];
                        sbyte sc2 = (sbyte)quantizedData[scBase + scIdx + 2];
                        sbyte sc4 = (sbyte)quantizedData[scBase + scIdx + 4];
                        sbyte sc6 = (sbyte)quantizedData[scBase + scIdx + 6];

                        output[outBase + l] = d * sc0 * q1;
                        output[outBase + l + 32] = d * sc2 * q2;
                        output[outBase + l + 64] = d * sc4 * q3;
                        output[outBase + l + 96] = d * sc6 * q4;
                    }
                }

                srcOffset += BytesPerBlock;
                dstOffset += BlockSize;
            }
        }

        /// <summary>
        /// Fused dequant + dot product for a single Q6_K row against a float vector.
        /// Avoids materializing the full dequantized row.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            float sum = 0f;
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    quantizedRow[srcOffset + DOffset],
                    quantizedRow[srcOffset + DOffset + 1]);

                int qlBase = srcOffset + QlOffset;
                int qhBase = srcOffset + QhOffset;
                int scBase = srcOffset + ScalesOffset;

                float blockSum = 0f;

                for (int half = 0; half < 2; half++)
                {
                    int qlOff = qlBase + half * 64;
                    int qhOff = qhBase + half * 32;
                    int scOff = half * 8;
                    int inBase = inputIdx + half * 128;

                    for (int l = 0; l < 32; l++)
                    {
                        int scIdx = scOff + l / 16;

                        int q1 = ((quantizedRow[qlOff + l] & 0x0F) |
                                  (((quantizedRow[qhOff + l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((quantizedRow[qlOff + l + 32] & 0x0F) |
                                  (((quantizedRow[qhOff + l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((quantizedRow[qlOff + l] >> 4) |
                                  (((quantizedRow[qhOff + l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((quantizedRow[qlOff + l + 32] >> 4) |
                                  (((quantizedRow[qhOff + l] >> 6) & 3) << 4)) - 32;

                        sbyte sc0 = (sbyte)quantizedRow[scBase + scIdx];
                        sbyte sc2 = (sbyte)quantizedRow[scBase + scIdx + 2];
                        sbyte sc4 = (sbyte)quantizedRow[scBase + scIdx + 4];
                        sbyte sc6 = (sbyte)quantizedRow[scBase + scIdx + 6];

                        blockSum += sc0 * q1 * input[inBase + l];
                        blockSum += sc2 * q2 * input[inBase + l + 32];
                        blockSum += sc4 * q3 * input[inBase + l + 64];
                        blockSum += sc6 * q4 * input[inBase + l + 96];
                    }
                }

                sum += d * blockSum;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return sum;
        }
    }
}
