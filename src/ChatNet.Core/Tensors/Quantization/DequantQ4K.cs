using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Q4_K dequantization: super-blocks of 256 values.
    /// Each block = 144 bytes. This is the format used by Q4_K_M quantization.
    ///
    /// Layout per super-block:
    ///   d[2]        : FP16 super-block scale
    ///   dmin[2]     : FP16 super-block min
    ///   scales[12]  : packed 6-bit sub-block scales and mins (8 sub-blocks of 32 elements)
    ///   qs[128]     : 256 x 4-bit quants packed (2 per byte, low/high nibble)
    ///
    /// Scale encoding (get_scale_min_k4 from llama.cpp):
    ///   For j=0..3: sc[j] = q[j] &amp; 63,  m[j] = q[j+4] &amp; 63
    ///   For j=4..7: sc[j] = (q[j+4] &amp; 0xF) | ((q[j-4] >> 6) &lt;&lt; 4)
    ///               m[j]  = (q[j+4] >> 4)  | ((q[j]   >> 6) &lt;&lt; 4)
    ///
    /// Dequantization per sub-block j (32 elements):
    ///   value = d * sc[j] * q4 - dmin * m[j]
    ///   where q4 is 0..15 (4-bit unsigned)
    /// </summary>
    public static class DequantQ4K
    {
        public const int BlockSize = 256;
        public const int BytesPerBlock = 144;

        private const int DOffset = 0;
        private const int DminOffset = 2;
        private const int ScalesOffset = 4;
        private const int QsOffset = 16;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Dequantize(ReadOnlySpan<byte> quantizedData, Span<float> output, int elementCount)
        {
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int dstOffset = 0;

            Span<int> sc = stackalloc int[8];
            Span<int> m = stackalloc int[8];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DOffset],
                    quantizedData[srcOffset + DOffset + 1]);
                float dmin = DequantQ4_0.HalfToFloat(
                    quantizedData[srcOffset + DminOffset],
                    quantizedData[srcOffset + DminOffset + 1]);

                DecodeScales(quantizedData, srcOffset + ScalesOffset, sc, m);

                int qsBase = srcOffset + QsOffset;

                for (int sub = 0; sub < 8; sub++)
                {
                    float dsc = d * sc[sub];
                    float dm = dmin * m[sub];
                    int qsByteStart = qsBase + sub * 16;
                    int outStart = dstOffset + sub * 32;

                    for (int j = 0; j < 16; j++)
                    {
                        byte qsByte = quantizedData[qsByteStart + j];
                        output[outStart + j] = dsc * (qsByte & 0x0F) - dm;
                        output[outStart + j + 16] = dsc * ((qsByte >> 4) & 0x0F) - dm;
                    }
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

            var acc = Vector128<float>.Zero;
            int* sc = stackalloc int[8];
            int* m = stackalloc int[8];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DOffset],
                    data[srcOffset + DOffset + 1]);
                float dmin = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DminOffset],
                    data[srcOffset + DminOffset + 1]);

                DecodeScalesUnsafe(data + srcOffset + ScalesOffset, sc, m);

                int qsBase = srcOffset + QsOffset;
                var vNibbleMask = Vector128.Create((byte)0x0F);
                var blockAcc = Vector128<float>.Zero;

                for (int sub = 0; sub < 8; sub++)
                {
                    float dsc = d * sc[sub];
                    float dm = dmin * m[sub];
                    var vDsc = Vector128.Create(dsc);
                    var vDm = Vector128.Create(dm);

                    int qsByteStart = qsBase + sub * 16;
                    int inBase = inputIdx + sub * 32;

                    var rawBytes = *(Vector128<byte>*)(data + qsByteStart);
                    var loNibbles = rawBytes & vNibbleMask;
                    var hiNibbles = Vector128.ShiftRightLogical(rawBytes.AsUInt16(), 4).AsByte() & vNibbleMask;

                    // Low nibbles: 16 values at positions 0..15
                    var loS0 = Vector128.WidenLower(loNibbles);
                    var loS1 = Vector128.WidenUpper(loNibbles);

                    var lf0 = Vector128.ConvertToSingle(Vector128.WidenLower(loS0).AsInt32());
                    var lf1 = Vector128.ConvertToSingle(Vector128.WidenUpper(loS0).AsInt32());
                    var lf2 = Vector128.ConvertToSingle(Vector128.WidenLower(loS1).AsInt32());
                    var lf3 = Vector128.ConvertToSingle(Vector128.WidenUpper(loS1).AsInt32());

                    blockAcc += (lf0 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 0);
                    blockAcc += (lf1 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 4);
                    blockAcc += (lf2 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 8);
                    blockAcc += (lf3 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 12);

                    // High nibbles: 16 values at positions 16..31
                    var hiS0 = Vector128.WidenLower(hiNibbles);
                    var hiS1 = Vector128.WidenUpper(hiNibbles);

                    var hf0 = Vector128.ConvertToSingle(Vector128.WidenLower(hiS0).AsInt32());
                    var hf1 = Vector128.ConvertToSingle(Vector128.WidenUpper(hiS0).AsInt32());
                    var hf2 = Vector128.ConvertToSingle(Vector128.WidenLower(hiS1).AsInt32());
                    var hf3 = Vector128.ConvertToSingle(Vector128.WidenUpper(hiS1).AsInt32());

                    blockAcc += (hf0 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 16);
                    blockAcc += (hf1 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 20);
                    blockAcc += (hf2 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 24);
                    blockAcc += (hf3 * vDsc - vDm) * *(Vector128<float>*)(input + inBase + 28);
                }

                acc += blockAcc;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return Vector128.Sum(acc);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float DotProductScalar(byte* data, float* input, int elementCount)
        {
            float sum = 0f;
            int blockCount = elementCount / BlockSize;
            int srcOffset = 0;
            int inputIdx = 0;

            int* sc = stackalloc int[8];
            int* m = stackalloc int[8];

            for (int b = 0; b < blockCount; b++)
            {
                float d = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DOffset],
                    data[srcOffset + DOffset + 1]);
                float dmin = DequantQ4_0.HalfToFloat(
                    data[srcOffset + DminOffset],
                    data[srcOffset + DminOffset + 1]);

                DecodeScalesUnsafe(data + srcOffset + ScalesOffset, sc, m);

                int qsBase = srcOffset + QsOffset;
                float blockSum = 0f;

                for (int sub = 0; sub < 8; sub++)
                {
                    float dsc = d * sc[sub];
                    float dm = dmin * m[sub];
                    int qsByteStart = qsBase + sub * 16;
                    int inBase = inputIdx + sub * 32;

                    for (int j = 0; j < 16; j++)
                    {
                        byte qsByte = data[qsByteStart + j];
                        blockSum += (dsc * (qsByte & 0x0F) - dm) * input[inBase + j];
                        blockSum += (dsc * ((qsByte >> 4) & 0x0F) - dm) * input[inBase + j + 16];
                    }
                }

                sum += blockSum;
                srcOffset += BytesPerBlock;
                inputIdx += BlockSize;
            }

            return sum;
        }

        /// <summary>
        /// Decode Q4_K / Q5_K packed scales (12 bytes) into 8 scale and 8 min values.
        /// Matches llama.cpp get_scale_min_k4.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void DecodeScales(ReadOnlySpan<byte> data, int offset, Span<int> sc, Span<int> m)
        {
            // j = 0..3: sc[j] = q[j] & 63, m[j] = q[j+4] & 63
            sc[0] = data[offset + 0] & 63;
            sc[1] = data[offset + 1] & 63;
            sc[2] = data[offset + 2] & 63;
            sc[3] = data[offset + 3] & 63;
            m[0] = data[offset + 4] & 63;
            m[1] = data[offset + 5] & 63;
            m[2] = data[offset + 6] & 63;
            m[3] = data[offset + 7] & 63;

            // j = 4..7: sc[j] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
            //           m[j]  = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4)
            sc[4] = (data[offset + 8] & 0xF) | ((data[offset + 0] >> 6) << 4);
            sc[5] = (data[offset + 9] & 0xF) | ((data[offset + 1] >> 6) << 4);
            sc[6] = (data[offset + 10] & 0xF) | ((data[offset + 2] >> 6) << 4);
            sc[7] = (data[offset + 11] & 0xF) | ((data[offset + 3] >> 6) << 4);

            m[4] = (data[offset + 8] >> 4) | ((data[offset + 4] >> 6) << 4);
            m[5] = (data[offset + 9] >> 4) | ((data[offset + 5] >> 6) << 4);
            m[6] = (data[offset + 10] >> 4) | ((data[offset + 6] >> 6) << 4);
            m[7] = (data[offset + 11] >> 4) | ((data[offset + 7] >> 6) << 4);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void DecodeScalesUnsafe(byte* q, int* sc, int* m)
        {
            sc[0] = q[0] & 63;
            sc[1] = q[1] & 63;
            sc[2] = q[2] & 63;
            sc[3] = q[3] & 63;
            m[0] = q[4] & 63;
            m[1] = q[5] & 63;
            m[2] = q[6] & 63;
            m[3] = q[7] & 63;

            sc[4] = (q[8] & 0xF) | ((q[0] >> 6) << 4);
            sc[5] = (q[9] & 0xF) | ((q[1] >> 6) << 4);
            sc[6] = (q[10] & 0xF) | ((q[2] >> 6) << 4);
            sc[7] = (q[11] & 0xF) | ((q[3] >> 6) << 4);

            m[4] = (q[8] >> 4) | ((q[4] >> 6) << 4);
            m[5] = (q[9] >> 4) | ((q[5] >> 6) << 4);
            m[6] = (q[10] >> 4) | ((q[6] >> 6) << 4);
            m[7] = (q[11] >> 4) | ((q[7] >> 6) << 4);
        }
    }
}
