using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q4_1 dequantization: blocks of 32 values.
    /// Each block = 2 bytes (half scale) + 2 bytes (half min) + 16 bytes (32 x 4-bit packed) = 20 bytes.
    ///
    /// Dequantization: value = quant * scale + min
    /// where quant is 0..15 (unsigned 4-bit).
    ///
    /// GGML packing order per block of 16 data bytes:
    ///   positions  0..15 = low  nibble of bytes 0..15  (qs[j] &amp; 0x0F)
    ///   positions 16..31 = high nibble of bytes 0..15  (qs[j] >> 4)
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
                int half = remaining < 16 ? remaining : 16;

                for (int j = 0; j < half; j++)
                {
                    int low = quantizedData[srcOffset + j] & 0x0F;
                    output[dstOffset + j] = low * scale + min;
                }

                int secondHalf = remaining < BlockSize ? (remaining > 16 ? remaining - 16 : 0) : 16;
                for (int j = 0; j < secondHalf; j++)
                {
                    int high = (quantizedData[srcOffset + j] >> 4) & 0x0F;
                    output[dstOffset + 16 + j] = high * scale + min;
                }

                srcOffset += 16;
                dstOffset += half + secondHalf;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(byte* quantizedRow, float* input, int elementCount)
        {
            if (Vector128.IsHardwareAccelerated)
                return DotProductVec128(quantizedRow, input, elementCount);

            return DotProductScalar(quantizedRow, input, elementCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotProductVec128(byte* quantizedRow, float* input, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            byte* src = quantizedRow;
            float* inp = input;

            var acc0 = Vector128<float>.Zero;
            var acc1 = Vector128<float>.Zero;
            var vNibbleMask = Vector128.Create((byte)0x0F);

            for (int b = 0; b < blockCount; b++)
            {
                float scale = DequantQ4_0.HalfToFloat(src[0], src[1]);
                src += 2;
                float min = DequantQ4_0.HalfToFloat(src[0], src[1]);
                src += 2;

                var vScale = Vector128.Create(scale);
                var vMin = Vector128.Create(min);

                var rawBytes = *(Vector128<byte>*)src;
                var loNibbles = rawBytes & vNibbleMask;
                var hiNibbles = Vector128.ShiftRightLogical(rawBytes.AsUInt16(), 4).AsByte() & vNibbleMask;

                var blockAcc0 = Vector128<float>.Zero;
                var blockAcc1 = Vector128<float>.Zero;

                // Low nibbles: positions 0..15
                var loS0 = Vector128.WidenLower(loNibbles);
                var loS1 = Vector128.WidenUpper(loNibbles);

                var lf0 = Vector128.ConvertToSingle(Vector128.WidenLower(loS0).AsInt32());
                var lf1 = Vector128.ConvertToSingle(Vector128.WidenUpper(loS0).AsInt32());
                var lf2 = Vector128.ConvertToSingle(Vector128.WidenLower(loS1).AsInt32());
                var lf3 = Vector128.ConvertToSingle(Vector128.WidenUpper(loS1).AsInt32());

                blockAcc0 += (lf0 * vScale + vMin) * *(Vector128<float>*)(inp + 0);
                blockAcc1 += (lf1 * vScale + vMin) * *(Vector128<float>*)(inp + 4);
                blockAcc0 += (lf2 * vScale + vMin) * *(Vector128<float>*)(inp + 8);
                blockAcc1 += (lf3 * vScale + vMin) * *(Vector128<float>*)(inp + 12);

                // High nibbles: positions 16..31
                var hiS0 = Vector128.WidenLower(hiNibbles);
                var hiS1 = Vector128.WidenUpper(hiNibbles);

                var hf0 = Vector128.ConvertToSingle(Vector128.WidenLower(hiS0).AsInt32());
                var hf1 = Vector128.ConvertToSingle(Vector128.WidenUpper(hiS0).AsInt32());
                var hf2 = Vector128.ConvertToSingle(Vector128.WidenLower(hiS1).AsInt32());
                var hf3 = Vector128.ConvertToSingle(Vector128.WidenUpper(hiS1).AsInt32());

                blockAcc0 += (hf0 * vScale + vMin) * *(Vector128<float>*)(inp + 16);
                blockAcc1 += (hf1 * vScale + vMin) * *(Vector128<float>*)(inp + 20);
                blockAcc0 += (hf2 * vScale + vMin) * *(Vector128<float>*)(inp + 24);
                blockAcc1 += (hf3 * vScale + vMin) * *(Vector128<float>*)(inp + 28);

                acc0 += blockAcc0;
                acc1 += blockAcc1;

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
                float scale = DequantQ4_0.HalfToFloat(quantizedRow[srcOffset], quantizedRow[srcOffset + 1]);
                srcOffset += 2;
                float min = DequantQ4_0.HalfToFloat(quantizedRow[srcOffset], quantizedRow[srcOffset + 1]);
                srcOffset += 2;

                float blockSum = 0f;
                float inputSum = 0f;

                for (int j = 0; j < 16; j++)
                {
                    int low = quantizedRow[srcOffset + j] & 0x0F;
                    blockSum += low * input[inputIdx + j];
                    inputSum += input[inputIdx + j];
                }

                for (int j = 0; j < 16; j++)
                {
                    int high = (quantizedRow[srcOffset + j] >> 4) & 0x0F;
                    blockSum += high * input[inputIdx + 16 + j];
                    inputSum += input[inputIdx + 16 + j];
                }

                sum += blockSum * scale + inputSum * min;
                srcOffset += 16;
                inputIdx += 32;
            }

            return sum;
        }
    }
}
