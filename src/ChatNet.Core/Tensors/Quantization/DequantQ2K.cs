using System;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q2_K dequantization: super-blocks of 256 values.
    /// Each block = 256/16*2 scales/mins packed + 256/4 quants + 2 d + 2 dmin = 84 bytes.
    ///
    /// Layout per super-block (84 bytes):
    ///   scales[16]  : packed 4-bit scale/min pairs (16 sub-blocks of 16 elements)
    ///   qs[64]      : 256 x 2-bit quants packed (4 per byte)
    ///   d[2]        : FP16 super-block scale
    ///   dmin[2]     : FP16 super-block min
    ///
    /// Dequantization per sub-block i (16 elements):
    ///   sc = (scales[i] &amp; 0xF) => sub-block scale nibble
    ///   m  = (scales[i] >> 4)   => sub-block min nibble
    ///   value = d * sc * q2 - dmin * m
    ///   where q2 is the 2-bit quant (0..3)
    /// </summary>
    public static class DequantQ2K
    {
        public const int BlockSize = 256;
        public const int BytesPerBlock = 84;

        private const int ScalesOffset = 0;
        private const int QsOffset = 16;
        private const int DOffset = 80;
        private const int DminOffset = 82;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
                float dmin = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DminOffset],
                    quantizedData[srcOffset + DminOffset + 1]);

                int scBase = srcOffset + ScalesOffset;
                int qsBase = srcOffset + QsOffset;

                // Q2_K layout: 2 groups of 128 elements, each group uses 32 qs bytes.
                // Within each group, 4 bit-shift iterations extract 2-bit values at shifts 0,2,4,6.
                // Each shift iteration produces 32 elements (2 sub-blocks of 16).
                // Scales are consumed 2 per shift iteration (16 total).
                int outIdx = dstOffset;
                int scaleIdx = 0;
                int qsGroupBase = qsBase;

                for (int group = 0; group < 2; group++)
                {
                    for (int shift = 0; shift < 8; shift += 2)
                    {
                        // First half: bytes [0..15] of this group, 16 elements
                        float dl = d * (quantizedData[scBase + scaleIdx] & 0x0F);
                        float ml = dmin * ((quantizedData[scBase + scaleIdx] >> 4) & 0x0F);
                        for (int l = 0; l < 16; l++)
                        {
                            output[outIdx++] = dl * ((quantizedData[qsGroupBase + l] >> shift) & 3) - ml;
                        }

                        // Second half: bytes [16..31] of this group, 16 elements
                        dl = d * (quantizedData[scBase + scaleIdx + 1] & 0x0F);
                        ml = dmin * ((quantizedData[scBase + scaleIdx + 1] >> 4) & 0x0F);
                        for (int l = 0; l < 16; l++)
                        {
                            output[outIdx++] = dl * ((quantizedData[qsGroupBase + 16 + l] >> shift) & 3) - ml;
                        }

                        scaleIdx += 2;
                    }
                    qsGroupBase += 32;
                }

                srcOffset += BytesPerBlock;
                dstOffset += BlockSize;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            return DotProductScalar(quantizedRow, input, elementCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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
                float dmin = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DminOffset],
                    data[srcOffset + DminOffset + 1]);

                int scBase = srcOffset + ScalesOffset;
                int qsBase = srcOffset + QsOffset;

                float blockSum = 0f;
                int yIdx = inputIdx;
                int scaleIdx = 0;
                int qsGroupBase = qsBase;

                for (int group = 0; group < 2; group++)
                {
                    for (int shift = 0; shift < 8; shift += 2)
                    {
                        int sc0 = data[scBase + scaleIdx] & 0x0F;
                        int m0 = (data[scBase + scaleIdx] >> 4) & 0x0F;
                        float dl0 = d * sc0;
                        float ml0 = dmin * m0;

                        for (int l = 0; l < 16; l++)
                        {
                            int q = (data[qsGroupBase + l] >> shift) & 3;
                            blockSum += (dl0 * q - ml0) * input[yIdx++];
                        }

                        int sc1 = data[scBase + scaleIdx + 1] & 0x0F;
                        int m1 = (data[scBase + scaleIdx + 1] >> 4) & 0x0F;
                        float dl1 = d * sc1;
                        float ml1 = dmin * m1;

                        for (int l = 0; l < 16; l++)
                        {
                            int q = (data[qsGroupBase + 16 + l] >> shift) & 3;
                            blockSum += (dl1 * q - ml1) * input[yIdx++];
                        }

                        scaleIdx += 2;
                    }
                    qsGroupBase += 32;
                }

                sum += blockSum;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return sum;
        }
    }
}
