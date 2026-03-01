using System;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q4_1 dequantization stub for future use.
    /// Q4_1: blocks of 32 values. Each block = 2 bytes (half scale) + 2 bytes (half min) + 16 bytes = 20 bytes.
    /// </summary>
    public static class DequantQ4_1
    {
        public const int BlockSize = 32;
        public const int BytesPerBlock = 20;

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
                float min = DequantQ4_0.HalfToFloat(quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                srcOffset += 2;

                int remaining = elementCount - dstOffset;
                int count = remaining < BlockSize ? remaining : BlockSize;

                for (int i = 0; i < count; i++)
                {
                    byte packed = quantizedData[srcOffset + i / 2];
                    int quant;
                    if ((i & 1) == 0)
                    {
                        quant = packed & 0x0F;
                    }
                    else
                    {
                        quant = (packed >> 4) & 0x0F;
                    }
                    output[dstOffset + i] = quant * scale + min;
                }

                srcOffset += 16;
                dstOffset += count;
            }
        }
    }
}
