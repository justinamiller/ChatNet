using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q5_K dequantization: super-blocks of 256 values.
    /// Each block = 176 bytes. This is the format used by Q5_K_M quantization.
    ///
    /// Layout per super-block:
    ///   d[2]        : FP16 super-block scale
    ///   dmin[2]     : FP16 super-block min
    ///   scales[12]  : packed 6-bit sub-block scales and mins (8 sub-blocks of 32 elements)
    ///   qh[32]      : 256 high bits (1 per element)
    ///   qs[128]     : 256 x 4-bit low quants packed (2 per byte)
    ///
    /// Scale encoding: same as Q4_K (get_scale_min_k4 from llama.cpp).
    ///
    /// Dequantization per sub-block j (32 elements):
    ///   value = d * sc[j] * q5 - dmin * m[j]
    ///   where q5 = low_nibble | (high_bit &lt;&lt; 4), range 0..31
    /// </summary>
    public static class DequantQ5K
    {
        public const int BlockSize = 256;
        public const int BytesPerBlock = 176;

        private const int DOffset = 0;
        private const int DminOffset = 2;
        private const int ScalesOffset = 4;
        private const int QhOffset = 16;
        private const int QsOffset = 48;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            Span<int> sc = stackalloc int[8];
            Span<int> m = stackalloc int[8];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DOffset],
                    quantizedData[srcOffset + DOffset + 1]);
                float dmin = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DminOffset],
                    quantizedData[srcOffset + DminOffset + 1]);

                DequantQ4K.DecodeScales(quantizedData, srcOffset + ScalesOffset, sc, m);

                int qhBase = srcOffset + QhOffset;
                int qsBase = srcOffset + QsOffset;

                for (int sub = 0; sub < 8; sub++)
                {
                    float dsc = d * sc[sub];
                    float dm = dmin * m[sub];
                    int qsByteStart = qsBase + sub * 16;
                    int outStart = dstOffset + sub * 32;

                    for (int j = 0; j < 16; j++)
                    {
                        int elemIdx = sub * 32 + j;
                        byte qsByte = quantizedData[qsByteStart + j];
                        int lowNib = qsByte & 0x0F;
                        int highBit = (quantizedData[qhBase + elemIdx / 8] >> (elemIdx % 8)) & 1;
                        int q5 = lowNib | (highBit << 4);
                        output[outStart + j] = dsc * q5 - dm;
                    }

                    for (int j = 0; j < 16; j++)
                    {
                        int elemIdx = sub * 32 + j + 16;
                        byte qsByte = quantizedData[qsByteStart + j];
                        int highNib = (qsByte >> 4) & 0x0F;
                        int highBit = (quantizedData[qhBase + elemIdx / 8] >> (elemIdx % 8)) & 1;
                        int q5 = highNib | (highBit << 4);
                        output[outStart + j + 16] = dsc * q5 - dm;
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

            int* sc = stackalloc int[8];
            int* m = stackalloc int[8];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DOffset],
                    data[srcOffset + DOffset + 1]);
                float dmin = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DminOffset],
                    data[srcOffset + DminOffset + 1]);

                DequantQ4K.DecodeScalesUnsafe(data + srcOffset + ScalesOffset, sc, m);

                int qhBase = srcOffset + QhOffset;
                int qsBase = srcOffset + QsOffset;
                float blockSum = 0f;

                for (int sub = 0; sub < 8; sub++)
                {
                    float dsc = d * sc[sub];
                    float dm = dmin * m[sub];
                    int qsByteStart = qsBase + sub * 16;
                    int inBase = inputIdx + sub * 32;

                    for (int j = 0; j < 16; j++)
                    {
                        int elemIdx = sub * 32 + j;
                        int lowNib = data[qsByteStart + j] & 0x0F;
                        int highBit = (data[qhBase + elemIdx / 8] >> (elemIdx % 8)) & 1;
                        int q5 = lowNib | (highBit << 4);
                        blockSum += (dsc * q5 - dm) * input[inBase + j];
                    }

                    for (int j = 0; j < 16; j++)
                    {
                        int elemIdx = sub * 32 + j + 16;
                        int highNib = (data[qsByteStart + j] >> 4) & 0x0F;
                        int highBit = (data[qhBase + elemIdx / 8] >> (elemIdx % 8)) & 1;
                        int q5 = highNib | (highBit << 4);
                        blockSum += (dsc * q5 - dm) * input[inBase + j + 16];
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
