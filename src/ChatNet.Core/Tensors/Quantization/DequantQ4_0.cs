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
                float scale = HalfToFloat(quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                srcOffset += 2;

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

                srcOffset += 16;
                dstOffset += half + secondHalf;
            }
        }

        /// <summary>
        /// Fused dequant + dot product for a single Q4_0 row against a float vector.
        /// Dispatches to SIMD (Vector128 byte-level widening) or scalar fallback.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            if (Vector128.IsHardwareAccelerated)
                return DotProductVec128(quantizedRow, input, elementCount);

            return DotProductScalar(quantizedRow, input, elementCount);
        }

        /// <summary>
        /// SIMD dot product using Vector128 byte-level widening.
        /// Loads 16 packed bytes at once, extracts nibbles via SIMD AND/shift,
        /// widens byte→ushort→uint→float, then vectorized multiply-accumulate.
        /// Uses dual accumulators for instruction-level parallelism.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotProductVec128(byte* quantizedRow, float* input, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            byte* src = quantizedRow;
            float* inp = input;

            var acc0 = Vector128<float>.Zero;
            var acc1 = Vector128<float>.Zero;
            var vNibbleMask = Vector128.Create((byte)0x0F);
            var vSub8 = Vector128.Create(8);

            for (int b = 0; b < blockCount; b++)
            {
                float scale = HalfToFloat(src[0], src[1]);
                src += 2;
                var vScale = Vector128.Create(scale);

                // Load 16 packed bytes as a single Vector128<byte>
                var rawBytes = *(Vector128<byte>*)src;

                // Extract low nibbles: AND with 0x0F (pure vector op)
                var loNibbles = rawBytes & vNibbleMask;

                // Extract high nibbles: 16-bit shift right 4 + mask (2 vector ops)
                var hiNibbles = Vector128.ShiftRightLogical(rawBytes.AsUInt16(), 4).AsByte() & vNibbleMask;

                var blockAcc0 = Vector128<float>.Zero;
                var blockAcc1 = Vector128<float>.Zero;

                // --- Low nibbles: 16 values at input positions 0..15 ---
                // Widen byte→ushort (2 halves of 8)
                var loS0 = Vector128.WidenLower(loNibbles);
                var loS1 = Vector128.WidenUpper(loNibbles);

                // Widen ushort→int32, subtract 8, convert to float (4 groups of 4)
                var lf0 = Vector128.ConvertToSingle(Vector128.WidenLower(loS0).AsInt32() - vSub8);
                var lf1 = Vector128.ConvertToSingle(Vector128.WidenUpper(loS0).AsInt32() - vSub8);
                var lf2 = Vector128.ConvertToSingle(Vector128.WidenLower(loS1).AsInt32() - vSub8);
                var lf3 = Vector128.ConvertToSingle(Vector128.WidenUpper(loS1).AsInt32() - vSub8);

                // Multiply-accumulate with input (dual accumulators for ILP)
                blockAcc0 += lf0 * *(Vector128<float>*)(inp + 0);
                blockAcc1 += lf1 * *(Vector128<float>*)(inp + 4);
                blockAcc0 += lf2 * *(Vector128<float>*)(inp + 8);
                blockAcc1 += lf3 * *(Vector128<float>*)(inp + 12);

                // --- High nibbles: 16 values at input positions 16..31 ---
                var hiS0 = Vector128.WidenLower(hiNibbles);
                var hiS1 = Vector128.WidenUpper(hiNibbles);

                var hf0 = Vector128.ConvertToSingle(Vector128.WidenLower(hiS0).AsInt32() - vSub8);
                var hf1 = Vector128.ConvertToSingle(Vector128.WidenUpper(hiS0).AsInt32() - vSub8);
                var hf2 = Vector128.ConvertToSingle(Vector128.WidenLower(hiS1).AsInt32() - vSub8);
                var hf3 = Vector128.ConvertToSingle(Vector128.WidenUpper(hiS1).AsInt32() - vSub8);

                blockAcc0 += hf0 * *(Vector128<float>*)(inp + 16);
                blockAcc1 += hf1 * *(Vector128<float>*)(inp + 20);
                blockAcc0 += hf2 * *(Vector128<float>*)(inp + 24);
                blockAcc1 += hf3 * *(Vector128<float>*)(inp + 28);

                // Apply per-block scale, merge into global accumulators
                acc0 += blockAcc0 * vScale;
                acc1 += blockAcc1 * vScale;

                src += 16;
                inp += 32;
            }

            return Vector128.Sum(acc0 + acc1);
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
