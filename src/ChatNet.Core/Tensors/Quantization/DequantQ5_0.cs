using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q5_0 dequantization: blocks of 32 values.
    /// Each block = 2 bytes (half scale) + 4 bytes (32 high-bits packed) + 16 bytes (32 x 4-bit low) = 22 bytes.
    ///
    /// Dequantization: value = (q5 - 16) * scale
    /// where q5 = low_nibble | (high_bit &lt;&lt; 4), range 0..31.
    ///
    /// Layout per block:
    ///   d[2]     : FP16 scale
    ///   qh[4]    : 32 high bits packed (1 bit per element)
    ///   qs[16]   : 32 low nibbles packed (2 per byte, low/high nibble order)
    /// </summary>
    public static class DequantQ5_0
    {
        public const int BlockSize = 32;
        public const int BytesPerBlock = 22;

        private const int DOffset = 0;
        private const int QhOffset = 2;
        private const int QsOffset = 6;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = (elementCount + BlockSize - 1) / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float scale = DequantQ4_0.HalfToFloat(quantizedData[srcOffset + DOffset], quantizedData[srcOffset + DOffset + 1]);

                // Read 32 high bits from qh[4]
                uint qh = (uint)(quantizedData[srcOffset + QhOffset]) |
                           ((uint)(quantizedData[srcOffset + QhOffset + 1]) << 8) |
                           ((uint)(quantizedData[srcOffset + QhOffset + 2]) << 16) |
                           ((uint)(quantizedData[srcOffset + QhOffset + 3]) << 24);

                int remaining = elementCount - dstOffset;
                int half = remaining < 16 ? remaining : 16;

                for (int j = 0; j < half; j++)
                {
                    int low = quantizedData[srcOffset + QsOffset + j] & 0x0F;
                    int highBit = (int)((qh >> j) & 1) << 4;
                    int q5 = low | highBit;
                    output[dstOffset + j] = (q5 - 16) * scale;
                }

                int secondHalf = remaining < BlockSize ? (remaining > 16 ? remaining - 16 : 0) : 16;
                for (int j = 0; j < secondHalf; j++)
                {
                    int high = (quantizedData[srcOffset + QsOffset + j] >> 4) & 0x0F;
                    int highBit = (int)((qh >> (j + 16)) & 1) << 4;
                    int q5 = high | highBit;
                    output[dstOffset + 16 + j] = (q5 - 16) * scale;
                }

                srcOffset += BytesPerBlock;
                dstOffset += half + secondHalf;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            return DotProductScalar(quantizedRow, input, elementCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotProductScalar(byte* quantizedRow, float* input, int elementCount)
        {
            float sum = 0f;
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float scale = DequantQ4_0.HalfToFloat(quantizedRow[srcOffset + DOffset], quantizedRow[srcOffset + DOffset + 1]);

                uint qh = *(uint*)(quantizedRow + srcOffset + QhOffset);

                float blockSum = 0f;

                for (int j = 0; j < 16; j++)
                {
                    int low = quantizedRow[srcOffset + QsOffset + j] & 0x0F;
                    int highBit = (int)((qh >> j) & 1) << 4;
                    int q5 = (low | highBit) - 16;
                    blockSum += q5 * input[inputIdx + j];
                }

                for (int j = 0; j < 16; j++)
                {
                    int high = (quantizedRow[srcOffset + QsOffset + j] >> 4) & 0x0F;
                    int highBit = (int)((qh >> (j + 16)) & 1) << 4;
                    int q5 = (high | highBit) - 16;
                    blockSum += q5 * input[inputIdx + 16 + j];
                }

                sum += blockSum * scale;
                srcOffset += BytesPerBlock;
                inputIdx += 32;
            }

            return sum;
        }
    }
}
