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

                for (int i = 0; i < 16; i++)
                {
                    int sc = quantizedData[scBase + i] & 0x0F;
                    int m = (quantizedData[scBase + i] >> 4) & 0x0F;
                    float dsc = d * sc;
                    float dm = dmin * m;

                    // Each sub-block has 16 elements, packed 4 per byte => 4 bytes per sub-block
                    int qsByteBase = qsBase + i * 4;
                    for (int j = 0; j < 4; j++)
                    {
                        byte qsByte = quantizedData[qsByteBase + j];
                        int outIdx = dstOffset + i * 16 + j * 4;
                        output[outIdx + 0] = dsc * ((qsByte >> 0) & 3) - dm;
                        output[outIdx + 1] = dsc * ((qsByte >> 2) & 3) - dm;
                        output[outIdx + 2] = dsc * ((qsByte >> 4) & 3) - dm;
                        output[outIdx + 3] = dsc * ((qsByte >> 6) & 3) - dm;
                    }
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

                for (int i = 0; i < 16; i++)
                {
                    int sc = data[scBase + i] & 0x0F;
                    int m = (data[scBase + i] >> 4) & 0x0F;
                    float dsc = d * sc;
                    float dm = dmin * m;

                    int qsByteBase = qsBase + i * 4;
                    int inBase = inputIdx + i * 16;
                    for (int j = 0; j < 4; j++)
                    {
                        byte qsByte = data[qsByteBase + j];
                        int inIdx = inBase + j * 4;
                        blockSum += (dsc * ((qsByte >> 0) & 3) - dm) * input[inIdx + 0];
                        blockSum += (dsc * ((qsByte >> 2) & 3) - dm) * input[inIdx + 1];
                        blockSum += (dsc * ((qsByte >> 4) & 3) - dm) * input[inIdx + 2];
                        blockSum += (dsc * ((qsByte >> 6) & 3) - dm) * input[inIdx + 3];
                    }
                }

                sum += blockSum;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return sum;
        }
    }
}
