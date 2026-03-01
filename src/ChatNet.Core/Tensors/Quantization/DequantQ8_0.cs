using System;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q8_0 dequantization stub for future use.
    /// Q8_0: blocks of 32 values. Each block = 2 bytes (half scale) + 32 bytes (32 x int8) = 34 bytes.
    /// </summary>
    public static class DequantQ8_0
    {
        public const int BlockSize = 32;
        public const int BytesPerBlock = 34;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = (elementCount + BlockSize - 1) / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float scale = DequantQ4_0.HalfToFloat(quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                srcOffset += 2;

                int remaining = elementCount - dstOffset;
                int count = remaining < BlockSize ? remaining : BlockSize;

                for (int i = 0; i < count; i++)
                {
                    output[dstOffset + i] = (sbyte)quantizedData[srcOffset + i] * scale;
                }

                srcOffset += 32;
                dstOffset += count;
            }
        }
    }
}
