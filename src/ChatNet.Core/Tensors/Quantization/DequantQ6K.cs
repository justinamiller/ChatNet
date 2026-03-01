using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

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

                int qlBase = srcOffset + QlOffset;
                int qhBase = srcOffset + QhOffset;
                int scBase = srcOffset + ScalesOffset;

                for (int half = 0; half < 2; half++)
                {
                    int qlOff = qlBase + half * 64;
                    int qhOff = qhBase + half * 32;
                    int scOff = half * 8;

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
        /// Uses Vector128 SIMD for the multiply-accumulate.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            if (Vector128.IsHardwareAccelerated)
                return DotProductVec128(quantizedRow, input, elementCount);

            return DotProductScalar(quantizedRow, input, elementCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotProductVec128(byte* data, float* input, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            var acc = Vector128<float>.Zero;

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DOffset],
                    data[srcOffset + DOffset + 1]);

                int qlBase = srcOffset + QlOffset;
                int qhBase = srcOffset + QhOffset;
                int scBase = srcOffset + ScalesOffset;

                var blockAcc = Vector128<float>.Zero;

                for (int half = 0; half < 2; half++)
                {
                    int qlOff = qlBase + half * 64;
                    int qhOff = qhBase + half * 32;
                    int scOff = half * 8;
                    int inBase = inputIdx + half * 128;

                    // Process l=0..15 (scale index = scOff+0) and l=16..31 (scale index = scOff+1)
                    for (int lBlock = 0; lBlock < 2; lBlock++)
                    {
                        int lStart = lBlock * 16;
                        int scIdx = scOff + lBlock;
                        float fSc0 = (sbyte)data[scBase + scIdx];
                        float fSc2 = (sbyte)data[scBase + scIdx + 2];
                        float fSc4 = (sbyte)data[scBase + scIdx + 4];
                        float fSc6 = (sbyte)data[scBase + scIdx + 6];
                        var vSc0 = Vector128.Create(fSc0);
                        var vSc2 = Vector128.Create(fSc2);
                        var vSc4 = Vector128.Create(fSc4);
                        var vSc6 = Vector128.Create(fSc6);

                        for (int l = lStart; l < lStart + 16; l += 4)
                        {
                            // q1 lane: elements at position l+0..l+3
                            var q1 = Vector128.Create(
                                (float)(((data[qlOff + l] & 0xF) | (((data[qhOff + l] >> 0) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 1] & 0xF) | (((data[qhOff + l + 1] >> 0) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 2] & 0xF) | (((data[qhOff + l + 2] >> 0) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 3] & 0xF) | (((data[qhOff + l + 3] >> 0) & 3) << 4)) - 32));
                            blockAcc += q1 * vSc0 * Vector128.Load(input + inBase + l);

                            // q2 lane: elements at position l+32..l+35
                            var q2 = Vector128.Create(
                                (float)(((data[qlOff + l + 32] & 0xF) | (((data[qhOff + l] >> 2) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 33] & 0xF) | (((data[qhOff + l + 1] >> 2) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 34] & 0xF) | (((data[qhOff + l + 2] >> 2) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 35] & 0xF) | (((data[qhOff + l + 3] >> 2) & 3) << 4)) - 32));
                            blockAcc += q2 * vSc2 * Vector128.Load(input + inBase + l + 32);

                            // q3 lane: elements at position l+64..l+67
                            var q3 = Vector128.Create(
                                (float)(((data[qlOff + l] >> 4) | (((data[qhOff + l] >> 4) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 1] >> 4) | (((data[qhOff + l + 1] >> 4) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 2] >> 4) | (((data[qhOff + l + 2] >> 4) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 3] >> 4) | (((data[qhOff + l + 3] >> 4) & 3) << 4)) - 32));
                            blockAcc += q3 * vSc4 * Vector128.Load(input + inBase + l + 64);

                            // q4 lane: elements at position l+96..l+99
                            var q4 = Vector128.Create(
                                (float)(((data[qlOff + l + 32] >> 4) | (((data[qhOff + l] >> 6) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 33] >> 4) | (((data[qhOff + l + 1] >> 6) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 34] >> 4) | (((data[qhOff + l + 2] >> 6) & 3) << 4)) - 32),
                                (float)(((data[qlOff + l + 35] >> 4) | (((data[qhOff + l + 3] >> 6) & 3) << 4)) - 32));
                            blockAcc += q4 * vSc6 * Vector128.Load(input + inBase + l + 96);
                        }
                    }
                }

                acc += blockAcc * Vector128.Create(d);
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return Vector128.Sum(acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotProductScalar(byte* data, float* input, int elementCount)
        {
            float sum = 0f;
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DOffset],
                    data[srcOffset + DOffset + 1]);

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

                        int q1 = ((data[qlOff + l] & 0x0F) |
                                  (((data[qhOff + l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((data[qlOff + l + 32] & 0x0F) |
                                  (((data[qhOff + l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((data[qlOff + l] >> 4) |
                                  (((data[qhOff + l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((data[qlOff + l + 32] >> 4) |
                                  (((data[qhOff + l] >> 6) & 3) << 4)) - 32;

                        sbyte sc0 = (sbyte)data[scBase + scIdx];
                        sbyte sc2 = (sbyte)data[scBase + scIdx + 2];
                        sbyte sc4 = (sbyte)data[scBase + scIdx + 4];
                        sbyte sc6 = (sbyte)data[scBase + scIdx + 6];

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
