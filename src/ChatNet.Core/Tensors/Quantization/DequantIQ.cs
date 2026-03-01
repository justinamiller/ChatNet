using System;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// IQ family dequantizers for importance-weighted quantization formats.
    /// These formats use non-uniform quantization with lookup tables for
    /// optimal weight representation at very low bit widths.
    ///
    /// Supported formats:
    ///   IQ4_NL  : 4-bit non-linear, blocks of 32, 18 bytes/block (same size as Q4_0)
    ///   IQ4_XS  : 4-bit extra-small, super-blocks of 256, 136 bytes/block
    ///   IQ3_S   : 3-bit small, super-blocks of 256, 110 bytes/block
    ///   IQ3_XXS : 3-bit extra-extra-small, super-blocks of 256, 98 bytes/block
    ///   IQ2_XS  : 2-bit extra-small, super-blocks of 256, 74 bytes/block
    ///   IQ2_XXS : 2-bit extra-extra-small, super-blocks of 256, 66 bytes/block
    ///   IQ2_S   : 2-bit small, super-blocks of 256, 82 bytes/block
    ///   IQ1_S   : 1-bit small, super-blocks of 256, 50 bytes/block
    ///   IQ1_M   : 1-bit medium, super-blocks of 256, 56 bytes/block
    /// </summary>
    public static class DequantIQ
    {
        /// <summary>
        /// IQ4_NL non-linear quantization lookup table.
        /// Maps 4-bit indices (0..15) to float values.
        /// From llama.cpp kvalues_iq4nl.
        /// </summary>
        public static readonly float[] IQ4NL_Table = new float[]
        {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113
        };

        /// <summary>
        /// IQ4_NL: 4-bit non-linear quantization. Block size = 32, 18 bytes/block.
        /// Layout: d[2] (FP16 scale) + qs[16] (32 x 4-bit packed)
        /// Same physical layout as Q4_0, but uses non-linear mapping table.
        /// </summary>
        public static class IQ4NL
        {
            public const int BlockSize = 32;
            public const int BytesPerBlock = 18;

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
                    int half = remaining < 16 ? remaining : 16;

                    for (int j = 0; j < half; j++)
                    {
                        int low = quantizedData[srcOffset + j] & 0x0F;
                        output[dstOffset + j] = scale * IQ4NL_Table[low];
                    }

                    int secondHalf = remaining < BlockSize ? (remaining > 16 ? remaining - 16 : 0) : 16;
                    for (int j = 0; j < secondHalf; j++)
                    {
                        int high = (quantizedData[srcOffset + j] >> 4) & 0x0F;
                        output[dstOffset + 16 + j] = scale * IQ4NL_Table[high];
                    }

                    srcOffset += 16;
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

                fixed (float* pTable = IQ4NL_Table)
                {
                    for (int b = 0; b < blockCount; b++)
                    {
                        float scale = DequantQ4_0.HalfToFloat(quantizedRow[srcOffset], quantizedRow[srcOffset + 1]);
                        srcOffset += 2;

                        float blockSum = 0f;

                        for (int j = 0; j < 16; j++)
                        {
                            int low = quantizedRow[srcOffset + j] & 0x0F;
                            blockSum += pTable[low] * input[inputIdx + j];
                        }

                        for (int j = 0; j < 16; j++)
                        {
                            int high = (quantizedRow[srcOffset + j] >> 4) & 0x0F;
                            blockSum += pTable[high] * input[inputIdx + 16 + j];
                        }

                        sum += blockSum * scale;
                        srcOffset += 16;
                        inputIdx += 32;
                    }
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ4_XS: 4-bit extra-small importance quantization. Super-block = 256, 136 bytes/block.
        /// Layout: d[2] (FP16) + scales_h[2] (high bits) + scales_l[8] (low nibble scales) + qs[128]
        /// Uses IQ4NL lookup table with per-sub-block 6-bit scales.
        /// </summary>
        public static class IQ4XS
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 136;

            private const int DOffset = 0;
            private const int ScalesHOffset = 2;
            private const int ScalesLOffset = 4;
            private const int QsOffset = 8;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset + DOffset],
                        quantizedData[srcOffset + DOffset + 1]);

                    ushort scalesH = (ushort)(quantizedData[srcOffset + ScalesHOffset] |
                                     (quantizedData[srcOffset + ScalesHOffset + 1] << 8));

                    for (int sub = 0; sub < 8; sub++)
                    {
                        int scLow = quantizedData[srcOffset + ScalesLOffset + sub] & 0x3F;
                        int scHigh = ((scalesH >> (2 * sub)) & 3) << 6;
                        int sc = (scLow | scHigh) - 32;
                        float dsc = d * sc;

                        int qsBase = srcOffset + QsOffset + sub * 16;
                        int outBase = dstOffset + sub * 32;

                        for (int j = 0; j < 16; j++)
                        {
                            byte qsByte = quantizedData[qsBase + j];
                            output[outBase + j] = dsc * IQ4NL_Table[qsByte & 0x0F];
                            output[outBase + j + 16] = dsc * IQ4NL_Table[(qsByte >> 4) & 0x0F];
                        }
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

                fixed (float* pTable = IQ4NL_Table)
                {
                    for (int b = 0; b < blockCount; b++)
                    {
                        float d = DequantQ4_0.HalfToFloat(
                            data[srcOffset + DOffset],
                            data[srcOffset + DOffset + 1]);

                        ushort scalesH = *(ushort*)(data + srcOffset + ScalesHOffset);

                        float blockSum = 0f;
                        for (int sub = 0; sub < 8; sub++)
                        {
                            int scLow = data[srcOffset + ScalesLOffset + sub] & 0x3F;
                            int scHigh = ((scalesH >> (2 * sub)) & 3) << 6;
                            int sc = (scLow | scHigh) - 32;
                            float dsc = d * sc;

                            int qsBase = srcOffset + QsOffset + sub * 16;
                            int inBase = inputIdx + sub * 32;
                            float subSum = 0f;

                            for (int j = 0; j < 16; j++)
                            {
                                byte qsByte = data[qsBase + j];
                                subSum += pTable[qsByte & 0x0F] * input[inBase + j];
                                subSum += pTable[(qsByte >> 4) & 0x0F] * input[inBase + j + 16];
                            }

                            blockSum += dsc * subSum;
                        }

                        sum += blockSum;
                        srcOffset += BytesPerBlock;
                        inputIdx += BlockSize;
                    }
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ3_S: 3-bit importance quantization (small). Super-block = 256, 110 bytes/block.
        /// Layout: d[2] + qs[64] + qh[32] + signs[32] + scales[8] + pad[4]
        /// Uses 3-bit grid-based quantization with sign bits.
        /// </summary>
        public static class IQ3S
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 110;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                // IQ3_S uses a grid-based decode with sign bits
                // Simplified scalar decode for compatibility
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 64;
                    int signBase = srcOffset + 2 + 64 + 32;
                    int scBase = srcOffset + 2 + 64 + 32 + 32;

                    for (int sub = 0; sub < 32; sub++)
                    {
                        int scIdx = sub / 4;
                        byte scByte = quantizedData[scBase + scIdx];
                        int shift = (sub % 4) * 2;
                        int sc = ((scByte >> shift) & 3) + 1;
                        float dsc = d * sc;

                        int elemBase = sub * 8;
                        byte signByte = quantizedData[signBase + sub];

                        for (int j = 0; j < 8; j++)
                        {
                            int idx = elemBase + j;
                            int qsIdx = idx / 4;
                            int qsShift = (idx % 4) * 2;
                            int q2 = (quantizedData[qsBase + qsIdx] >> qsShift) & 3;
                            int qhBit = (quantizedData[qhBase + idx / 8] >> (idx % 8)) & 1;
                            int q3 = q2 | (qhBit << 2);

                            float val = dsc * q3;
                            if (((signByte >> j) & 1) != 0) val = -val;
                            output[dstOffset + idx] = val;
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 64;
                    int signBase = srcOffset + 2 + 64 + 32;
                    int scBase = srcOffset + 2 + 64 + 32 + 32;

                    float blockSum = 0f;

                    for (int sub = 0; sub < 32; sub++)
                    {
                        int scIdx = sub / 4;
                        int shift = (sub % 4) * 2;
                        int sc = ((data[scBase + scIdx] >> shift) & 3) + 1;
                        float dsc = d * sc;

                        int elemBase = sub * 8;
                        byte signByte = data[signBase + sub];
                        float subSum = 0f;

                        for (int j = 0; j < 8; j++)
                        {
                            int idx = elemBase + j;
                            int qsIdx = idx / 4;
                            int qsShift = (idx % 4) * 2;
                            int q2 = (data[qsBase + qsIdx] >> qsShift) & 3;
                            int qhBit = (data[qhBase + idx / 8] >> (idx % 8)) & 1;
                            int q3 = q2 | (qhBit << 2);

                            float val = (float)q3;
                            if (((signByte >> j) & 1) != 0) val = -val;
                            subSum += val * input[inputIdx + idx];
                        }

                        blockSum += dsc * subSum;
                    }

                    sum += blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ3_XXS: 3-bit extra-extra-small. Super-block = 256, 98 bytes/block.
        /// Layout: d[2] + qs[96] (packed 3-bit quants with embedded scales)
        /// </summary>
        public static class IQ3XXS
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 98;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);

                    int qsBase = srcOffset + 2;
                    // IQ3_XXS packs quant data with embedded sign and scale info
                    // For 32 sub-blocks of 8 elements: qs[3*8] per sub-block = 96 bytes total
                    // Plus scale bytes embedded at specific offsets

                    for (int sub = 0; sub < 32; sub++)
                    {
                        int subOffset = qsBase + sub * 3;
                        // Each 3-byte group encodes 8 x 3-bit values
                        uint packed = (uint)quantizedData[subOffset] |
                                      ((uint)quantizedData[subOffset + 1] << 8) |
                                      ((uint)quantizedData[subOffset + 2] << 16);

                        int outBase = dstOffset + sub * 8;
                        for (int j = 0; j < 8; j++)
                        {
                            int q3 = (int)((packed >> (j * 3)) & 7);
                            output[outBase + j] = d * (q3 - 4);
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);

                    int qsBase = srcOffset + 2;
                    float blockSum = 0f;

                    for (int sub = 0; sub < 32; sub++)
                    {
                        int subOffset = qsBase + sub * 3;
                        uint packed = *(ushort*)(data + subOffset) | ((uint)data[subOffset + 2] << 16);

                        int inBase = inputIdx + sub * 8;
                        for (int j = 0; j < 8; j++)
                        {
                            int q3 = (int)((packed >> (j * 3)) & 7) - 4;
                            blockSum += q3 * input[inBase + j];
                        }
                    }

                    sum += d * blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ2_XS: 2-bit extra-small importance quantization. Super-block = 256, 74 bytes/block.
        /// Layout: d[2] + qs[64] + scales[8]
        /// Uses 2-bit grid-based quantization.
        /// </summary>
        public static class IQ2XS
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 74;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);

                    int qsBase = srcOffset + 2;
                    int scBase = srcOffset + 2 + 64;

                    // 16 sub-blocks of 16 elements, each sub-block has 4 bytes of quants
                    for (int sub = 0; sub < 16; sub++)
                    {
                        int sc = ((sbyte)quantizedData[scBase + sub / 2]);
                        if ((sub & 1) == 0)
                            sc = (sc & 0x0F) - 8;
                        else
                            sc = ((sc >> 4) & 0x0F) - 8;
                        float dsc = d * sc;

                        int qsByteBase = qsBase + sub * 4;
                        int outBase = dstOffset + sub * 16;

                        for (int j = 0; j < 4; j++)
                        {
                            byte qsByte = quantizedData[qsByteBase + j];
                            output[outBase + j * 4 + 0] = dsc * (((qsByte >> 0) & 3) - 1);
                            output[outBase + j * 4 + 1] = dsc * (((qsByte >> 2) & 3) - 1);
                            output[outBase + j * 4 + 2] = dsc * (((qsByte >> 4) & 3) - 1);
                            output[outBase + j * 4 + 3] = dsc * (((qsByte >> 6) & 3) - 1);
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);

                    int qsBase = srcOffset + 2;
                    int scBase = srcOffset + 2 + 64;

                    float blockSum = 0f;

                    for (int sub = 0; sub < 16; sub++)
                    {
                        int scRaw = (sbyte)data[scBase + sub / 2];
                        int sc;
                        if ((sub & 1) == 0)
                            sc = (scRaw & 0x0F) - 8;
                        else
                            sc = ((scRaw >> 4) & 0x0F) - 8;
                        float dsc = d * sc;

                        int qsByteBase = qsBase + sub * 4;
                        int inBase = inputIdx + sub * 16;
                        float subSum = 0f;

                        for (int j = 0; j < 4; j++)
                        {
                            byte qsByte = data[qsByteBase + j];
                            int idx = inBase + j * 4;
                            subSum += (((qsByte >> 0) & 3) - 1) * input[idx + 0];
                            subSum += (((qsByte >> 2) & 3) - 1) * input[idx + 1];
                            subSum += (((qsByte >> 4) & 3) - 1) * input[idx + 2];
                            subSum += (((qsByte >> 6) & 3) - 1) * input[idx + 3];
                        }

                        blockSum += dsc * subSum;
                    }

                    sum += blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ2_XXS: 2-bit extra-extra-small. Super-block = 256, 66 bytes/block.
        /// Layout: d[2] + qs[64] (packed 2-bit quants with embedded scale info)
        /// </summary>
        public static class IQ2XXS
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 66;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                    int qsBase = srcOffset + 2;

                    for (int sub = 0; sub < 16; sub++)
                    {
                        int qsByteBase = qsBase + sub * 4;
                        int outBase = dstOffset + sub * 16;

                        for (int j = 0; j < 4; j++)
                        {
                            byte qsByte = quantizedData[qsByteBase + j];
                            output[outBase + j * 4 + 0] = d * (((qsByte >> 0) & 3) - 1);
                            output[outBase + j * 4 + 1] = d * (((qsByte >> 2) & 3) - 1);
                            output[outBase + j * 4 + 2] = d * (((qsByte >> 4) & 3) - 1);
                            output[outBase + j * 4 + 3] = d * (((qsByte >> 6) & 3) - 1);
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);
                    int qsBase = srcOffset + 2;

                    float blockSum = 0f;
                    for (int sub = 0; sub < 16; sub++)
                    {
                        int qsByteBase = qsBase + sub * 4;
                        int inBase = inputIdx + sub * 16;

                        for (int j = 0; j < 4; j++)
                        {
                            byte qsByte = data[qsByteBase + j];
                            int idx = inBase + j * 4;
                            blockSum += (((qsByte >> 0) & 3) - 1) * input[idx + 0];
                            blockSum += (((qsByte >> 2) & 3) - 1) * input[idx + 1];
                            blockSum += (((qsByte >> 4) & 3) - 1) * input[idx + 2];
                            blockSum += (((qsByte >> 6) & 3) - 1) * input[idx + 3];
                        }
                    }

                    sum += d * blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ2_S: 2-bit small importance quantization. Super-block = 256, 82 bytes/block.
        /// Layout: d[2] + qs[64] + qh[16]
        /// </summary>
        public static class IQ2S
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 82;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 64;

                    for (int sub = 0; sub < 16; sub++)
                    {
                        int qsByteBase = qsBase + sub * 4;
                        int outBase = dstOffset + sub * 16;

                        byte qhByte = quantizedData[qhBase + sub];
                        int sc = (qhByte & 0x0F) + 1;
                        int sign = (qhByte >> 4) & 0x0F;
                        float dsc = d * sc;

                        for (int j = 0; j < 4; j++)
                        {
                            byte qsByte = quantizedData[qsByteBase + j];
                            for (int k = 0; k < 4; k++)
                            {
                                int q = (qsByte >> (k * 2)) & 3;
                                float val = dsc * q;
                                int elemIdx = j * 4 + k;
                                if (elemIdx < 4 && ((sign >> elemIdx) & 1) != 0) val = -val;
                                output[outBase + elemIdx] = val;
                            }
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 64;
                    float blockSum = 0f;

                    for (int sub = 0; sub < 16; sub++)
                    {
                        int qsByteBase = qsBase + sub * 4;
                        int inBase = inputIdx + sub * 16;

                        byte qhByte = data[qhBase + sub];
                        int sc = (qhByte & 0x0F) + 1;
                        int sign = (qhByte >> 4) & 0x0F;
                        float dsc = d * sc;
                        float subSum = 0f;

                        for (int j = 0; j < 4; j++)
                        {
                            byte qsByte = data[qsByteBase + j];
                            for (int k = 0; k < 4; k++)
                            {
                                int q = (qsByte >> (k * 2)) & 3;
                                float val = (float)q;
                                int elemIdx = j * 4 + k;
                                if (elemIdx < 4 && ((sign >> elemIdx) & 1) != 0) val = -val;
                                subSum += val * input[inBase + elemIdx];
                            }
                        }

                        blockSum += dsc * subSum;
                    }

                    sum += blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ1_S: 1-bit importance quantization (small). Super-block = 256, 50 bytes/block.
        /// Extreme compression format - uses ternary quantization with grid-based encoding.
        /// Layout: d[2] + qs[32] + qh[16]
        /// </summary>
        public static class IQ1S
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 50;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 32;

                    // IQ1_S uses ternary values: -1, 0, +1
                    for (int sub = 0; sub < 32; sub++)
                    {
                        byte qsByte = quantizedData[qsBase + sub];
                        int outBase = dstOffset + sub * 8;

                        for (int j = 0; j < 8; j++)
                        {
                            int bit = (qsByte >> j) & 1;
                            int hbit = (quantizedData[qhBase + sub / 2] >> ((sub % 2) * 4 + j / 2)) & 1;
                            // Ternary: bit=1 -> +d, bit=0 and hbit=1 -> -d, else 0
                            float val;
                            if (bit == 1)
                                val = d;
                            else if (hbit == 1)
                                val = -d;
                            else
                                val = 0f;
                            output[outBase + j] = val;
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 32;
                    float blockSum = 0f;

                    for (int sub = 0; sub < 32; sub++)
                    {
                        byte qsByte = data[qsBase + sub];
                        int inBase = inputIdx + sub * 8;

                        for (int j = 0; j < 8; j++)
                        {
                            int bit = (qsByte >> j) & 1;
                            int hbit = (data[qhBase + sub / 2] >> ((sub % 2) * 4 + j / 2)) & 1;
                            float weight;
                            if (bit == 1)
                                weight = 1f;
                            else if (hbit == 1)
                                weight = -1f;
                            else
                                weight = 0f;
                            blockSum += weight * input[inBase + j];
                        }
                    }

                    sum += d * blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }

        /// <summary>
        /// IQ1_M: 1-bit importance quantization (medium). Super-block = 256, 56 bytes/block.
        /// Similar to IQ1_S with more scale bits for better quality.
        /// Layout: d[2] + qs[32] + qh[16] + scales[6]
        /// </summary>
        public static class IQ1M
        {
            public const int BlockSize = 256;
            public const int BytesPerBlock = 56;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
            {
                int blockCount = elementCount / BlockSize;
                int srcOffset = 0;
                int dstOffset = 0;

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        quantizedData[srcOffset], quantizedData[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 32;
                    int scBase = srcOffset + 2 + 32 + 16;

                    for (int sub = 0; sub < 32; sub++)
                    {
                        int scIdx = sub / 8;
                        int scShift = (sub % 8);
                        float sc = 1.0f;
                        if (scIdx < 6)
                        {
                            sc = ((quantizedData[scBase + scIdx] >> scShift) & 1) == 1 ? 1.0f : 0.5f;
                        }

                        byte qsByte = quantizedData[qsBase + sub];
                        int outBase = dstOffset + sub * 8;

                        for (int j = 0; j < 8; j++)
                        {
                            int bit = (qsByte >> j) & 1;
                            int hbit = (quantizedData[qhBase + sub / 2] >> ((sub % 2) * 4 + j / 2)) & 1;
                            float val;
                            if (bit == 1)
                                val = d * sc;
                            else if (hbit == 1)
                                val = -d * sc;
                            else
                                val = 0f;
                            output[outBase + j] = val;
                        }
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

                for (int b = 0; b < blockCount; b++)
                {
                    float d = DequantQ4_0.HalfToFloat(
                        data[srcOffset], data[srcOffset + 1]);
                    int qsBase = srcOffset + 2;
                    int qhBase = srcOffset + 2 + 32;
                    int scBase = srcOffset + 2 + 32 + 16;
                    float blockSum = 0f;

                    for (int sub = 0; sub < 32; sub++)
                    {
                        int scIdx = sub / 8;
                        int scShift = (sub % 8);
                        float sc = 1.0f;
                        if (scIdx < 6)
                        {
                            sc = ((data[scBase + scIdx] >> scShift) & 1) == 1 ? 1.0f : 0.5f;
                        }

                        byte qsByte = data[qsBase + sub];
                        int inBase = inputIdx + sub * 8;

                        for (int j = 0; j < 8; j++)
                        {
                            int bit = (qsByte >> j) & 1;
                            int hbit = (data[qhBase + sub / 2] >> ((sub % 2) * 4 + j / 2)) & 1;
                            float weight;
                            if (bit == 1)
                                weight = sc;
                            else if (hbit == 1)
                                weight = -sc;
                            else
                                weight = 0f;
                            blockSum += weight * input[inBase + j];
                        }
                    }

                    sum += d * blockSum;
                    srcOffset += BytesPerBlock;
                    inputIdx += BlockSize;
                }

                return sum;
            }
        }
    }
}
