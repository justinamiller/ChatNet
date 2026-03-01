using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q4_0 dequantization: blocks of 32 values.
    /// Each block = 2 bytes (half-float scale) + 16 bytes (32 x 4-bit packed) = 18 bytes.
    ///
    /// GGML packing order per block of 16 data bytes:
    ///   positions  0..15 = low  nibble of bytes 0..15  (qs[j] &amp; 0x0F)
    ///   positions 16..31 = high nibble of bytes 0..15  (qs[j] >> 4)
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

                // Unpack 16 bytes into 32 float values using GGML nibble order:
                //   first 16 values  from low nibbles  (qs[j] & 0x0F) - 8
                //   next  16 values  from high nibbles (qs[j] >> 4)   - 8
                int remaining = elementCount - dstOffset;
                int half = remaining < 16 ? remaining : 16;

                for (int j = 0; j < half; j++)
                {
                    int low = (quantizedData[srcOffset + j] & 0x0F) - 8;
                    output[dstOffset + j] = low * scale;
                }

                int secondHalf = remaining < BlockSize ? (remaining > 16 ? remaining - 16 : 0) : 16;
                for (int j = 0; j < secondHalf; j++)
                {
                    int high = ((quantizedData[srcOffset + j] >> 4) & 0x0F) - 8;
                    output[dstOffset + 16 + j] = high * scale;
                }

                srcOffset += 16; // 16 bytes of packed data per block
                dstOffset += half + secondHalf;
            }
        }

        /// <summary>
        /// Fused dequant + dot product for a single Q4_0 row against a float vector.
        /// Uses Vector128 SIMD (ARM NEON / x86 SSE) for the multiply-accumulate.
        ///
        /// GGML nibble order: low nibbles → positions 0..15, high nibbles → positions 16..31.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            if (Vector128.IsHardwareAccelerated)
                return DotProductVec128(quantizedRow, input, elementCount);

            return DotProductScalar(quantizedRow, input, elementCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotProductVec128(byte* quantizedRow, float* input, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            byte* src = quantizedRow;
            float* inp = input;

            var acc = Vector128<float>.Zero;

            for (int b = 0; b < blockCount; b++)
            {
                float scale = HalfToFloat(src[0], src[1]);
                src += 2;
                var vScale = Vector128.Create(scale);

                var blockAcc = Vector128<float>.Zero;

                // Low nibbles: 16 values at positions 0..15, process 4 at a time
                for (int j = 0; j < 16; j += 4)
                {
                    var q = Vector128.Create(
                        (float)((src[j] & 0xF) - 8),
                        (float)((src[j + 1] & 0xF) - 8),
                        (float)((src[j + 2] & 0xF) - 8),
                        (float)((src[j + 3] & 0xF) - 8));
                    var v = Vector128.Load(inp + j);
                    blockAcc += q * v;
                }

                // High nibbles: 16 values at positions 16..31, process 4 at a time
                for (int j = 0; j < 16; j += 4)
                {
                    var q = Vector128.Create(
                        (float)((src[j] >> 4) - 8),
                        (float)((src[j + 1] >> 4) - 8),
                        (float)((src[j + 2] >> 4) - 8),
                        (float)((src[j + 3] >> 4) - 8));
                    var v = Vector128.Load(inp + 16 + j);
                    blockAcc += q * v;
                }

                acc += blockAcc * vScale;
                src += 16;
                inp += 32;
            }

            return Vector128.Sum(acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float DotProductScalar(byte* quantizedRow, float* input, int elementCount)
        {
            float sum = 0f;
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float scale = HalfToFloat(quantizedRow[srcOffset], quantizedRow[srcOffset + 1]);
                srcOffset += 2;

                float blockSum = 0f;

                for (int j = 0; j < 16; j++)
                {
                    int low = (quantizedRow[srcOffset + j] & 0x0F) - 8;
                    blockSum += low * input[inputIdx + j];
                }

                for (int j = 0; j < 16; j++)
                {
                    int high = ((quantizedRow[srcOffset + j] >> 4) & 0x0F) - 8;
                    blockSum += high * input[inputIdx + 16 + j];
                }

                sum += blockSum * scale;
                srcOffset += 16;
                inputIdx += 32;
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
