using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q4_0 dequantization: blocks of 32 values.
    /// Each block = 2 bytes (half-float scale) + 16 bytes (32 x 4-bit packed) = 18 bytes.
    /// </summary>
    public static class DequantQ4_0
    {
        /// <summary>Block size for Q4_0: 32 elements per block.</summary>
        public const int BlockSize = 32;

        /// <summary>Bytes per block: 2 (scale) + 16 (data) = 18.</summary>
        public const int BytesPerBlock = 18;

        /// <summary>
        /// Dequantize Q4_0 data into a float output buffer.
        /// </summary>
        /// <param name="quantizedData">Raw Q4_0 bytes.</param>
        /// <param name="output">Float output buffer (must be large enough for all elements).</param>
        /// <param name="elementCount">Number of elements to dequantize.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = (elementCount + BlockSize - 1) / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            for (int b = 0; b < blockCount; b++)
            {
                // Read half-float scale (2 bytes, little-endian)
                float scale = HalfToFloat(quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                srcOffset += 2;

                // Read 16 bytes of packed 4-bit values (32 values)
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
                    // Center around zero: subtract 8
                    output[dstOffset + i] = (quant - 8) * scale;
                }

                srcOffset += 16; // 16 bytes of packed data per block
                dstOffset += count;
            }
        }

        /// <summary>
        /// Fused dequant + dot product for a single Q4_0 row against a float vector.
        /// This is the hottest path - avoids materializing the full dequantized row.
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
                // Read half-float scale
                float scale = HalfToFloat(quantizedRow[srcOffset], quantizedRow[srcOffset + 1]);
                srcOffset += 2;

                // Process 32 elements: 16 bytes of packed data
                float blockSum = 0f;
                for (int i = 0; i < 16; i++)
                {
                    byte packed = quantizedRow[srcOffset + i];
                    int low = packed & 0x0F;
                    int high = (packed >> 4) & 0x0F;
                    blockSum += (low - 8) * input[inputIdx];
                    blockSum += (high - 8) * input[inputIdx + 1];
                    inputIdx += 2;
                }
                sum += blockSum * scale;
                srcOffset += 16;
            }

            return sum;
        }

        /// <summary>
        /// Convert IEEE 754 half-precision float (2 bytes LE) to float.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HalfToFloat(byte lo, byte hi)
        {
            ushort h = (ushort)(lo | (hi << 8));
            return (float)BitConverter.UInt16BitsToHalf(h);
        }
    }
}
