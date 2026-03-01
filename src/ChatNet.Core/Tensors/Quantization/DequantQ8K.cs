using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q8_K dequantization: super-blocks of 256 values.
    /// Each block = 292 bytes.
    ///
    /// Layout per super-block:
    ///   d[4]        : float32 super-block scale
    ///   qs[256]     : 256 x int8 quants
    ///   bsums[32]   : 16 x int16 block sums (used by other quant types for mixed-quant ops)
    ///
    /// Dequantization: value = d * (sbyte)qs[i]
    /// </summary>
    public static class DequantQ8K
    {
        public const int BlockSize = 256;
        public const int BytesPerBlock = 292;

        private const int DOffset = 0;
        private const int QsOffset = 4;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            for (int b = 0; b < blockCount; b++)
            {
                float d = BitConverter.Int32BitsToSingle(
                    quantizedData[srcOffset + DOffset] |
                    (quantizedData[srcOffset + DOffset + 1] << 8) |
                    (quantizedData[srcOffset + DOffset + 2] << 16) |
                    (quantizedData[srcOffset + DOffset + 3] << 24));

                int qsBase = srcOffset + QsOffset;

                for (int i = 0; i < BlockSize; i++)
                {
                    output[dstOffset + i] = (sbyte)quantizedData[qsBase + i] * d;
                }

                srcOffset += BytesPerBlock;
                dstOffset += BlockSize;
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
        private static unsafe float DotProductVec128(byte* data, float* input, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            var acc0 = Vector128<float>.Zero;
            var acc1 = Vector128<float>.Zero;

            for (int b = 0; b < blockCount; b++)
            {
                float d = *(float*)(data + srcOffset + DOffset);
                var vScale = Vector128.Create(d);
                int qsBase = srcOffset + QsOffset;

                // Process 256 int8 values in groups of 8
                for (int i = 0; i < BlockSize; i += 8)
                {
                    var v0 = Vector128.Create(
                        (float)(sbyte)data[qsBase + i],
                        (float)(sbyte)data[qsBase + i + 1],
                        (float)(sbyte)data[qsBase + i + 2],
                        (float)(sbyte)data[qsBase + i + 3]);
                    var v1 = Vector128.Create(
                        (float)(sbyte)data[qsBase + i + 4],
                        (float)(sbyte)data[qsBase + i + 5],
                        (float)(sbyte)data[qsBase + i + 6],
                        (float)(sbyte)data[qsBase + i + 7]);

                    acc0 += v0 * vScale * *(Vector128<float>*)(input + inputIdx + i);
                    acc1 += v1 * vScale * *(Vector128<float>*)(input + inputIdx + i + 4);
                }

                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return Vector128.Sum(acc0 + acc1);
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
                float d = *(float*)(data + srcOffset + DOffset);
                int qsBase = srcOffset + QsOffset;

                float blockSum = 0f;
                for (int i = 0; i < BlockSize; i++)
                {
                    blockSum += (sbyte)data[qsBase + i] * input[inputIdx + i];
                }

                sum += blockSum * d;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return sum;
        }
    }
}
