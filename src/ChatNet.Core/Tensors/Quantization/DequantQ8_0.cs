using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q8_0 dequantization: blocks of 32 values.
    /// Each block = 2 bytes (half scale) + 32 bytes (32 x int8) = 34 bytes.
    ///
    /// Dequantization: value = (sbyte)qs[i] * scale
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

            for (int b = 0; b < blockCount; b++)
            {
                float scale = DequantQ4_0.HalfToFloat(src[0], src[1]);
                src += 2;
                var vScale = Vector128.Create(scale);

                // Process 32 int8 values in groups of 8 using SIMD
                for (int i = 0; i < 32; i += 8)
                {
                    var v0 = Vector128.Create(
                        (float)(sbyte)src[i],
                        (float)(sbyte)src[i + 1],
                        (float)(sbyte)src[i + 2],
                        (float)(sbyte)src[i + 3]);
                    var v1 = Vector128.Create(
                        (float)(sbyte)src[i + 4],
                        (float)(sbyte)src[i + 5],
                        (float)(sbyte)src[i + 6],
                        (float)(sbyte)src[i + 7]);

                    acc0 += v0 * vScale * *(Vector128<float>*)(inp + i);
                    acc1 += v1 * vScale * *(Vector128<float>*)(inp + i + 4);
                }

                src += 32;
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

                float blockSum = 0f;

                for (int i = 0; i < 32; i++)
                {
                    blockSum += (sbyte)quantizedRow[srcOffset + i] * input[inputIdx + i];
                }

                sum += blockSum * scale;
                srcOffset += 32;
                inputIdx += 32;
            }

            return sum;
        }
    }
}
