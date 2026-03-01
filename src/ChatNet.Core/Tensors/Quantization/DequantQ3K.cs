using System;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q3_K dequantization: super-blocks of 256 values.
    /// Each block = 110 bytes.
    ///
    /// Layout per super-block:
    ///   hmask[32]   : high bits of 3-bit quants (256 bits, 1 per element)
    ///   qs[64]      : low 2 bits of quants (2 bits per element, 4 per byte)
    ///   scales[12]  : packed 6-bit sub-block scales (16 sub-blocks of 16 elements)
    ///   d[2]        : FP16 super-block scale
    ///
    /// Scale encoding (from llama.cpp):
    ///   bytes 0..7: low 4 bits of 16 scale values (low nibble = sc[0..7], high nibble = sc[8..15])
    ///   bytes 8..11: high 2 bits of all 16 scales (4 bit-pairs per byte)
    ///   Each decoded value is unsigned 6-bit, then subtract 32 for signed.
    ///
    /// Dequantization: value = d * scale * (q3 - 4)
    /// where q3 = (2-bit from qs) | (high_bit &lt;&lt; 2), range 0..7
    /// </summary>
    public static class DequantQ3K
    {
        public const int BlockSize = 256;
        public const int BytesPerBlock = 110;

        private const int HmaskOffset = 0;
        private const int QsOffset = 32;
        private const int ScalesOffset = 96;
        private const int DOffset = 108;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            Span<int> scales = stackalloc int[16];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DOffset],
                    quantizedData[srcOffset + DOffset + 1]);

                DecodeScales(quantizedData, srcOffset + ScalesOffset, scales);

                int hmBase = srcOffset + HmaskOffset;
                int qsBase = srcOffset + QsOffset;

                for (int i = 0; i < 256; i++)
                {
                    int subBlock = i / 16;
                    int qsByte = quantizedData[qsBase + i / 4];
                    int shift = (i % 4) * 2;
                    int q2 = (qsByte >> shift) & 3;

                    int hmByte = quantizedData[hmBase + i / 8];
                    int hmBit = (hmByte >> (i % 8)) & 1;
                    int q3 = q2 | (hmBit << 2);

                    output[dstOffset + i] = d * scales[subBlock] * (q3 - 4);
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

            int* scales = stackalloc int[16];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DOffset],
                    data[srcOffset + DOffset + 1]);

                DecodeScalesUnsafe(data + srcOffset + ScalesOffset, scales);

                int hmBase = srcOffset + HmaskOffset;
                int qsBase = srcOffset + QsOffset;

                float blockSum = 0f;

                for (int i = 0; i < 256; i++)
                {
                    int subBlock = i / 16;
                    int q2 = (data[qsBase + i / 4] >> ((i % 4) * 2)) & 3;
                    int hmBit = (data[hmBase + i / 8] >> (i % 8)) & 1;
                    int q3 = q2 | (hmBit << 2);

                    blockSum += scales[subBlock] * (q3 - 4) * input[inputIdx + i];
                }

                sum += d * blockSum;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return sum;
        }

        /// <summary>
        /// Decode 12 bytes of packed scales into 16 signed 6-bit scale values.
        /// Matches llama.cpp Q3_K scale decode.
        /// </summary>
        private static void DecodeScales(ReadOnlySpan<byte> data, int offset, Span<int> scales)
        {
            // Low 4 bits: bytes 0..7 low nibbles -> sc[0..7], high nibbles -> sc[8..15]
            for (int i = 0; i < 8; i++)
            {
                scales[i] = data[offset + i] & 0x0F;
                scales[i + 8] = (data[offset + i] >> 4) & 0x0F;
            }

            // High 2 bits from bytes 8..11
            // bytes 8..11 each carry 4 pairs of 2 bits for scales 0..3, 4..7, 8..11, 12..15
            for (int i = 0; i < 4; i++)
            {
                byte hb = data[offset + 8 + i];
                scales[i + 0] |= ((hb >> 0) & 3) << 4;
                scales[i + 4] |= ((hb >> 2) & 3) << 4;
                scales[i + 8] |= ((hb >> 4) & 3) << 4;
                scales[i + 12] |= ((hb >> 6) & 3) << 4;
            }

            // Convert from unsigned 6-bit to signed
            for (int i = 0; i < 16; i++)
            {
                scales[i] -= 32;
            }
        }

        private static unsafe void DecodeScalesUnsafe(byte* data, int* scales)
        {
            for (int i = 0; i < 8; i++)
            {
                scales[i] = data[i] & 0x0F;
                scales[i + 8] = (data[i] >> 4) & 0x0F;
            }

            for (int i = 0; i < 4; i++)
            {
                byte hb = data[8 + i];
                scales[i + 0] |= ((hb >> 0) & 3) << 4;
                scales[i + 4] |= ((hb >> 2) & 3) << 4;
                scales[i + 8] |= ((hb >> 4) & 3) << 4;
                scales[i + 12] |= ((hb >> 6) & 3) << 4;
            }

            for (int i = 0; i < 16; i++)
            {
                scales[i] -= 32;
            }
        }
    }
}
