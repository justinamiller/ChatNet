using System;
using System.Runtime.InteropServices;
using ChatNet.Core.Tensors.Quantization;
using ChatNet.Core.Gguf;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ChatNet.Tests
{
    /// <summary>
    /// Comprehensive tests for all quantization/dequantization formats.
    /// Tests verify round-trip correctness by encoding known float values into
    /// quantized format, then dequantizing and checking the results.
    /// Also tests fused DotProduct implementations against scalar reference.
    /// </summary>
    [TestClass]
    public class QuantizationTests
    {
        private const float Tolerance = 0.01f;
        private const float LooseTolerance = 0.1f;

        #region Q4_0 Tests

        [TestMethod]
        public void Q4_0_Dequantize_ProducesCorrectValues()
        {
            // Build a Q4_0 block: 2 bytes scale (FP16) + 16 bytes data = 18 bytes
            // Scale = 1.0, data = all zeros (which means q=0, value = (0-8)*1 = -8)
            byte[] data = new byte[DequantQ4_0.BytesPerBlock];
            WriteHalf(data, 0, 1.0f);
            // qs = all 0x00 => low nibble = 0, high nibble = 0
            // value = (0 - 8) * 1.0 = -8.0

            float[] output = new float[DequantQ4_0.BlockSize];
            DequantQ4_0.Dequantize(data, output, DequantQ4_0.BlockSize);

            for (int i = 0; i < DequantQ4_0.BlockSize; i++)
            {
                Assert.AreEqual(-8.0f, output[i], Tolerance,
                    $"Q4_0 element {i} mismatch");
            }
        }

        [TestMethod]
        public void Q4_0_Dequantize_MidpointValues()
        {
            byte[] data = new byte[DequantQ4_0.BytesPerBlock];
            WriteHalf(data, 0, 0.5f);
            // Fill data with 0x88 => low nibble = 8, high nibble = 8
            // value = (8 - 8) * 0.5 = 0.0
            for (int i = 2; i < 18; i++) data[i] = 0x88;

            float[] output = new float[DequantQ4_0.BlockSize];
            DequantQ4_0.Dequantize(data, output, DequantQ4_0.BlockSize);

            for (int i = 0; i < DequantQ4_0.BlockSize; i++)
            {
                Assert.AreEqual(0.0f, output[i], Tolerance,
                    $"Q4_0 midpoint element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q4_0_DotProduct_MatchesScalar()
        {
            byte[] qData = new byte[DequantQ4_0.BytesPerBlock];
            WriteHalf(qData, 0, 2.0f);
            // Fill with alternating nibbles: 0x37 => low=7, high=3
            for (int i = 2; i < 18; i++) qData[i] = 0x37;

            float[] inputVec = new float[DequantQ4_0.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            // Compute expected via dequantize
            float[] dequantized = new float[DequantQ4_0.BlockSize];
            DequantQ4_0.Dequantize(qData, dequantized, DequantQ4_0.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ4_0.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ4_0.DotProduct(pData, pInput, DequantQ4_0.BlockSize);
                Assert.AreEqual(expected, actual, Tolerance,
                    "Q4_0 DotProduct mismatch");
            }
        }

        #endregion

        #region Q4_1 Tests

        [TestMethod]
        public void Q4_1_Dequantize_ProducesCorrectValues()
        {
            byte[] data = new byte[DequantQ4_1.BytesPerBlock];
            WriteHalf(data, 0, 2.0f); // scale
            WriteHalf(data, 2, 1.0f); // min
            // qs = all 0x00 => q=0, value = 0 * 2 + 1 = 1.0

            float[] output = new float[DequantQ4_1.BlockSize];
            DequantQ4_1.Dequantize(data, output, DequantQ4_1.BlockSize);

            for (int i = 0; i < DequantQ4_1.BlockSize; i++)
            {
                Assert.AreEqual(1.0f, output[i], Tolerance,
                    $"Q4_1 element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q4_1_DotProduct_MatchesScalar()
        {
            byte[] qData = new byte[DequantQ4_1.BytesPerBlock];
            WriteHalf(qData, 0, 1.0f); // scale
            WriteHalf(qData, 2, 0.5f); // min
            for (int i = 4; i < 20; i++) qData[i] = 0x55; // nibbles = 5

            float[] inputVec = new float[DequantQ4_1.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ4_1.BlockSize];
            DequantQ4_1.Dequantize(qData, dequantized, DequantQ4_1.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ4_1.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ4_1.DotProduct(pData, pInput, DequantQ4_1.BlockSize);
                Assert.AreEqual(expected, actual, Tolerance,
                    "Q4_1 DotProduct mismatch");
            }
        }

        #endregion

        #region Q5_0 Tests

        [TestMethod]
        public void Q5_0_Dequantize_ZeroQuants()
        {
            byte[] data = new byte[DequantQ5_0.BytesPerBlock];
            WriteHalf(data, 0, 1.0f); // scale
            // qh = 0, qs = 0 => q5 = 0, value = (0 - 16) * 1 = -16

            float[] output = new float[DequantQ5_0.BlockSize];
            DequantQ5_0.Dequantize(data, output, DequantQ5_0.BlockSize);

            for (int i = 0; i < DequantQ5_0.BlockSize; i++)
            {
                Assert.AreEqual(-16.0f, output[i], Tolerance,
                    $"Q5_0 element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q5_0_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ5_0.BytesPerBlock];
            WriteHalf(qData, 0, 0.5f); // scale
            // Set some high bits and quant data
            qData[2] = 0xFF; qData[3] = 0xFF; qData[4] = 0x00; qData[5] = 0x00; // qh
            for (int i = 6; i < 22; i++) qData[i] = 0xAA; // alternating nibbles

            float[] inputVec = new float[DequantQ5_0.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 0.5f;

            float[] dequantized = new float[DequantQ5_0.BlockSize];
            DequantQ5_0.Dequantize(qData, dequantized, DequantQ5_0.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ5_0.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ5_0.DotProduct(pData, pInput, DequantQ5_0.BlockSize);
                Assert.AreEqual(expected, actual, Tolerance,
                    "Q5_0 DotProduct mismatch");
            }
        }

        #endregion

        #region Q5_1 Tests

        [TestMethod]
        public void Q5_1_Dequantize_WithMinOffset()
        {
            byte[] data = new byte[DequantQ5_1.BytesPerBlock];
            WriteHalf(data, 0, 1.0f); // scale
            WriteHalf(data, 2, 2.0f); // min
            // qh = 0, qs = 0 => q5 = 0, value = 0 * 1 + 2 = 2.0

            float[] output = new float[DequantQ5_1.BlockSize];
            DequantQ5_1.Dequantize(data, output, DequantQ5_1.BlockSize);

            for (int i = 0; i < DequantQ5_1.BlockSize; i++)
            {
                Assert.AreEqual(2.0f, output[i], Tolerance,
                    $"Q5_1 element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q5_1_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ5_1.BytesPerBlock];
            WriteHalf(qData, 0, 0.25f); // scale
            WriteHalf(qData, 2, 0.5f);  // min
            for (int i = 8; i < 24; i++) qData[i] = 0x33; // nibbles

            float[] inputVec = new float[DequantQ5_1.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ5_1.BlockSize];
            DequantQ5_1.Dequantize(qData, dequantized, DequantQ5_1.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ5_1.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ5_1.DotProduct(pData, pInput, DequantQ5_1.BlockSize);
                Assert.AreEqual(expected, actual, Tolerance,
                    "Q5_1 DotProduct mismatch");
            }
        }

        #endregion

        #region Q8_0 Tests

        [TestMethod]
        public void Q8_0_Dequantize_ProducesCorrectValues()
        {
            byte[] data = new byte[DequantQ8_0.BytesPerBlock];
            WriteHalf(data, 0, 0.5f); // scale
            // qs = all 10 (as sbyte) => value = 10 * 0.5 = 5.0
            for (int i = 2; i < 34; i++) data[i] = 10;

            float[] output = new float[DequantQ8_0.BlockSize];
            DequantQ8_0.Dequantize(data, output, DequantQ8_0.BlockSize);

            for (int i = 0; i < DequantQ8_0.BlockSize; i++)
            {
                Assert.AreEqual(5.0f, output[i], Tolerance,
                    $"Q8_0 element {i} mismatch");
            }
        }

        [TestMethod]
        public void Q8_0_Dequantize_NegativeValues()
        {
            byte[] data = new byte[DequantQ8_0.BytesPerBlock];
            WriteHalf(data, 0, 1.0f); // scale
            // qs = all 0xF6 = -10 as sbyte => value = -10 * 1.0 = -10.0
            for (int i = 2; i < 34; i++) data[i] = 0xF6;

            float[] output = new float[DequantQ8_0.BlockSize];
            DequantQ8_0.Dequantize(data, output, DequantQ8_0.BlockSize);

            for (int i = 0; i < DequantQ8_0.BlockSize; i++)
            {
                Assert.AreEqual(-10.0f, output[i], Tolerance,
                    $"Q8_0 negative element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q8_0_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ8_0.BytesPerBlock];
            WriteHalf(qData, 0, 0.25f); // scale
            for (int i = 2; i < 34; i++) qData[i] = (byte)(i - 2); // 0..31 as sbyte

            float[] inputVec = new float[DequantQ8_0.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ8_0.BlockSize];
            DequantQ8_0.Dequantize(qData, dequantized, DequantQ8_0.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ8_0.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ8_0.DotProduct(pData, pInput, DequantQ8_0.BlockSize);
                Assert.AreEqual(expected, actual, Tolerance,
                    "Q8_0 DotProduct mismatch");
            }
        }

        #endregion

        #region Q6_K Tests

        [TestMethod]
        public void Q6K_Dequantize_ProducesFiniteValues()
        {
            // Q6_K has complex layout, just verify no NaN/Inf
            byte[] data = new byte[DequantQ6K.BytesPerBlock];
            WriteHalf(data, DequantQ6K.BytesPerBlock - 2, 1.0f); // d at end of block
            // Set some scale values
            for (int i = 192; i < 208; i++) data[i] = 1;

            float[] output = new float[DequantQ6K.BlockSize];
            DequantQ6K.Dequantize(data, output, DequantQ6K.BlockSize);

            for (int i = 0; i < DequantQ6K.BlockSize; i++)
            {
                Assert.IsFalse(float.IsNaN(output[i]), $"Q6_K element {i} is NaN");
                Assert.IsFalse(float.IsInfinity(output[i]), $"Q6_K element {i} is Inf");
            }
        }

        [TestMethod]
        public unsafe void Q6K_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ6K.BytesPerBlock];
            WriteHalf(qData, DequantQ6K.BytesPerBlock - 2, 0.1f); // d
            for (int i = 192; i < 208; i++) qData[i] = 2; // scales
            for (int i = 0; i < 128; i++) qData[i] = 0x22;

            float[] inputVec = new float[DequantQ6K.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ6K.BlockSize];
            DequantQ6K.Dequantize(qData, dequantized, DequantQ6K.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ6K.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ6K.DotProduct(pData, pInput, DequantQ6K.BlockSize);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "Q6_K DotProduct mismatch");
            }
        }

        #endregion

        #region Q2_K Tests

        [TestMethod]
        public void Q2K_Dequantize_ProducesFiniteValues()
        {
            byte[] data = new byte[DequantQ2K.BytesPerBlock];
            WriteHalf(data, 80, 1.0f); // d
            WriteHalf(data, 82, 0.5f); // dmin
            for (int i = 0; i < 16; i++) data[i] = 0x21; // scale=1, min=2

            float[] output = new float[DequantQ2K.BlockSize];
            DequantQ2K.Dequantize(data, output, DequantQ2K.BlockSize);

            for (int i = 0; i < DequantQ2K.BlockSize; i++)
            {
                Assert.IsFalse(float.IsNaN(output[i]), $"Q2_K element {i} is NaN");
                Assert.IsFalse(float.IsInfinity(output[i]), $"Q2_K element {i} is Inf");
            }
        }

        [TestMethod]
        public unsafe void Q2K_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ2K.BytesPerBlock];
            WriteHalf(qData, 80, 0.5f); // d
            WriteHalf(qData, 82, 0.25f); // dmin
            for (int i = 0; i < 16; i++) qData[i] = 0x11;
            for (int i = 16; i < 80; i++) qData[i] = 0x55;

            float[] inputVec = new float[DequantQ2K.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ2K.BlockSize];
            DequantQ2K.Dequantize(qData, dequantized, DequantQ2K.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ2K.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ2K.DotProduct(pData, pInput, DequantQ2K.BlockSize);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "Q2_K DotProduct mismatch");
            }
        }

        #endregion

        #region Q3_K Tests

        [TestMethod]
        public void Q3K_Dequantize_ProducesFiniteValues()
        {
            byte[] data = new byte[DequantQ3K.BytesPerBlock];
            WriteHalf(data, 108, 1.0f); // d

            float[] output = new float[DequantQ3K.BlockSize];
            DequantQ3K.Dequantize(data, output, DequantQ3K.BlockSize);

            for (int i = 0; i < DequantQ3K.BlockSize; i++)
            {
                Assert.IsFalse(float.IsNaN(output[i]), $"Q3_K element {i} is NaN");
                Assert.IsFalse(float.IsInfinity(output[i]), $"Q3_K element {i} is Inf");
            }
        }

        [TestMethod]
        public unsafe void Q3K_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ3K.BytesPerBlock];
            WriteHalf(qData, 108, 0.1f); // d
            for (int i = 32; i < 96; i++) qData[i] = 0x55;

            float[] inputVec = new float[DequantQ3K.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ3K.BlockSize];
            DequantQ3K.Dequantize(qData, dequantized, DequantQ3K.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ3K.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ3K.DotProduct(pData, pInput, DequantQ3K.BlockSize);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "Q3_K DotProduct mismatch");
            }
        }

        #endregion

        #region Q4_K Tests (Q4_K_M)

        [TestMethod]
        public void Q4K_Dequantize_ProducesFiniteValues()
        {
            byte[] data = new byte[DequantQ4K.BytesPerBlock];
            WriteHalf(data, 0, 1.0f); // d
            WriteHalf(data, 2, 0.5f); // dmin
            // Set some scales
            for (int i = 4; i < 16; i++) data[i] = 0x11;

            float[] output = new float[DequantQ4K.BlockSize];
            DequantQ4K.Dequantize(data, output, DequantQ4K.BlockSize);

            for (int i = 0; i < DequantQ4K.BlockSize; i++)
            {
                Assert.IsFalse(float.IsNaN(output[i]), $"Q4_K element {i} is NaN");
                Assert.IsFalse(float.IsInfinity(output[i]), $"Q4_K element {i} is Inf");
            }
        }

        [TestMethod]
        public unsafe void Q4K_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ4K.BytesPerBlock];
            WriteHalf(qData, 0, 0.5f); // d
            WriteHalf(qData, 2, 0.25f); // dmin
            for (int i = 4; i < 16; i++) qData[i] = 0x11;
            for (int i = 16; i < 144; i++) qData[i] = 0x33;

            float[] inputVec = new float[DequantQ4K.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ4K.BlockSize];
            DequantQ4K.Dequantize(qData, dequantized, DequantQ4K.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ4K.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ4K.DotProduct(pData, pInput, DequantQ4K.BlockSize);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "Q4_K DotProduct mismatch");
            }
        }

        #endregion

        #region Q5_K Tests (Q5_K_M)

        [TestMethod]
        public void Q5K_Dequantize_ProducesFiniteValues()
        {
            byte[] data = new byte[DequantQ5K.BytesPerBlock];
            WriteHalf(data, 0, 1.0f); // d
            WriteHalf(data, 2, 0.5f); // dmin
            for (int i = 4; i < 16; i++) data[i] = 0x11;

            float[] output = new float[DequantQ5K.BlockSize];
            DequantQ5K.Dequantize(data, output, DequantQ5K.BlockSize);

            for (int i = 0; i < DequantQ5K.BlockSize; i++)
            {
                Assert.IsFalse(float.IsNaN(output[i]), $"Q5_K element {i} is NaN");
                Assert.IsFalse(float.IsInfinity(output[i]), $"Q5_K element {i} is Inf");
            }
        }

        [TestMethod]
        public unsafe void Q5K_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ5K.BytesPerBlock];
            WriteHalf(qData, 0, 0.5f); // d
            WriteHalf(qData, 2, 0.25f); // dmin
            for (int i = 4; i < 16; i++) qData[i] = 0x11;
            for (int i = 48; i < 176; i++) qData[i] = 0x55;

            float[] inputVec = new float[DequantQ5K.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ5K.BlockSize];
            DequantQ5K.Dequantize(qData, dequantized, DequantQ5K.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ5K.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ5K.DotProduct(pData, pInput, DequantQ5K.BlockSize);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "Q5_K DotProduct mismatch");
            }
        }

        #endregion

        #region Q8_K Tests

        [TestMethod]
        public void Q8K_Dequantize_ProducesCorrectValues()
        {
            byte[] data = new byte[DequantQ8K.BytesPerBlock];
            // Write float32 scale = 0.5
            byte[] scaleBytes = BitConverter.GetBytes(0.5f);
            Array.Copy(scaleBytes, 0, data, 0, 4);
            // qs = all 10 => value = 0.5 * 10 = 5.0
            for (int i = 4; i < 260; i++) data[i] = 10;

            float[] output = new float[DequantQ8K.BlockSize];
            DequantQ8K.Dequantize(data, output, DequantQ8K.BlockSize);

            for (int i = 0; i < DequantQ8K.BlockSize; i++)
            {
                Assert.AreEqual(5.0f, output[i], Tolerance,
                    $"Q8_K element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q8K_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantQ8K.BytesPerBlock];
            byte[] scaleBytes = BitConverter.GetBytes(0.25f);
            Array.Copy(scaleBytes, 0, qData, 0, 4);
            for (int i = 4; i < 260; i++) qData[i] = (byte)(i % 20);

            float[] inputVec = new float[DequantQ8K.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantQ8K.BlockSize];
            DequantQ8K.Dequantize(qData, dequantized, DequantQ8K.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantQ8K.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ8K.DotProduct(pData, pInput, DequantQ8K.BlockSize);
                Assert.AreEqual(expected, actual, Tolerance,
                    "Q8_K DotProduct mismatch");
            }
        }

        #endregion

        #region IQ4_NL Tests

        [TestMethod]
        public void IQ4NL_Dequantize_UsesLookupTable()
        {
            byte[] data = new byte[DequantIQ.IQ4NL.BytesPerBlock];
            WriteHalf(data, 0, 1.0f); // scale
            // Set first data byte to 0x80 => low nibble=0, high nibble=8
            data[2] = 0x80;

            float[] output = new float[DequantIQ.IQ4NL.BlockSize];
            DequantIQ.IQ4NL.Dequantize(data, output, DequantIQ.IQ4NL.BlockSize);

            // Element 0: table[0] * 1.0 = -127
            Assert.AreEqual(DequantIQ.IQ4NL_Table[0], output[0], Tolerance,
                "IQ4_NL table lookup for index 0");
            // Element 16: table[8] * 1.0 = 1
            Assert.AreEqual(DequantIQ.IQ4NL_Table[8], output[16], Tolerance,
                "IQ4_NL table lookup for index 8");
        }

        [TestMethod]
        public unsafe void IQ4NL_DotProduct_MatchesDequantize()
        {
            byte[] qData = new byte[DequantIQ.IQ4NL.BytesPerBlock];
            WriteHalf(qData, 0, 0.01f); // small scale
            for (int i = 2; i < 18; i++) qData[i] = 0x88; // all index 8

            float[] inputVec = new float[DequantIQ.IQ4NL.BlockSize];
            for (int i = 0; i < inputVec.Length; i++) inputVec[i] = 1.0f;

            float[] dequantized = new float[DequantIQ.IQ4NL.BlockSize];
            DequantIQ.IQ4NL.Dequantize(qData, dequantized, DequantIQ.IQ4NL.BlockSize);
            float expected = 0f;
            for (int i = 0; i < DequantIQ.IQ4NL.BlockSize; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantIQ.IQ4NL.DotProduct(pData, pInput, DequantIQ.IQ4NL.BlockSize);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "IQ4_NL DotProduct mismatch");
            }
        }

        #endregion

        #region GgufTensorInfo ByteSize Tests

        [TestMethod]
        public void GgufTensorInfo_ByteSize_AllTypes()
        {
            // Verify byte size calculation for each quantization type
            var testCases = new (GgmlType type, int blockSize, int bytesPerBlock)[]
            {
                (GgmlType.F32, 1, 4),
                (GgmlType.F16, 1, 2),
                (GgmlType.Q4_0, 32, 18),
                (GgmlType.Q4_1, 32, 20),
                (GgmlType.Q5_0, 32, 22),
                (GgmlType.Q5_1, 32, 24),
                (GgmlType.Q8_0, 32, 34),
                (GgmlType.Q2K, 256, 84),
                (GgmlType.Q3K, 256, 110),
                (GgmlType.Q4K, 256, 144),
                (GgmlType.Q5K, 256, 176),
                (GgmlType.Q6K, 256, 210),
                (GgmlType.Q8K, 256, 292),
                (GgmlType.IQ4NL, 32, 18),
                (GgmlType.IQ4XS, 256, 136),
                (GgmlType.IQ3S, 256, 110),
                (GgmlType.IQ3XXS, 256, 98),
                (GgmlType.IQ2XS, 256, 74),
                (GgmlType.IQ2XXS, 256, 66),
                (GgmlType.IQ2S, 256, 82),
                (GgmlType.IQ1S, 256, 50),
                (GgmlType.IQ1M, 256, 56),
            };

            foreach (var (type, blockSize, bytesPerBlock) in testCases)
            {
                int elements = blockSize == 1 ? 1024 : blockSize * 4; // 4 blocks
                var info = new GgufTensorInfo
                {
                    Name = $"test_{type}",
                    NDimensions = 1,
                    Dimensions = new ulong[] { (ulong)elements },
                    Type = type,
                    Offset = 0
                };

                ulong expectedSize;
                if (blockSize == 1)
                {
                    expectedSize = (ulong)(elements * bytesPerBlock);
                }
                else
                {
                    ulong blocks = (ulong)((elements + blockSize - 1) / blockSize);
                    expectedSize = blocks * (ulong)bytesPerBlock;
                }

                Assert.AreEqual(expectedSize, info.ByteSize,
                    $"ByteSize mismatch for {type}: expected {expectedSize}, got {info.ByteSize}");
            }
        }

        #endregion

        #region Multi-Block Tests

        [TestMethod]
        public void Q4_0_MultiBlock_Dequantize()
        {
            int elements = 64; // 2 blocks
            byte[] data = new byte[2 * DequantQ4_0.BytesPerBlock];

            // Block 1: scale=1.0, data=0x88 (q=8, value=0)
            WriteHalf(data, 0, 1.0f);
            for (int i = 2; i < 18; i++) data[i] = 0x88;

            // Block 2: scale=2.0, data=0x88 (q=8, value=0)
            WriteHalf(data, 18, 2.0f);
            for (int i = 20; i < 36; i++) data[i] = 0x88;

            float[] output = new float[elements];
            DequantQ4_0.Dequantize(data, output, elements);

            for (int i = 0; i < elements; i++)
            {
                Assert.AreEqual(0.0f, output[i], Tolerance,
                    $"Q4_0 multi-block element {i} mismatch");
            }
        }

        [TestMethod]
        public unsafe void Q4K_MultiBlock_DotProduct()
        {
            int elements = 512; // 2 blocks
            byte[] qData = new byte[2 * DequantQ4K.BytesPerBlock];

            for (int block = 0; block < 2; block++)
            {
                int off = block * DequantQ4K.BytesPerBlock;
                WriteHalf(qData, off, 0.5f);     // d
                WriteHalf(qData, off + 2, 0.25f); // dmin
                for (int i = off + 4; i < off + 16; i++) qData[i] = 0x11;
                for (int i = off + 16; i < off + 144; i++) qData[i] = 0x55;
            }

            float[] inputVec = new float[elements];
            for (int i = 0; i < elements; i++) inputVec[i] = 0.5f;

            float[] dequantized = new float[elements];
            DequantQ4K.Dequantize(qData, dequantized, elements);
            float expected = 0f;
            for (int i = 0; i < elements; i++)
                expected += dequantized[i] * inputVec[i];

            fixed (byte* pData = qData)
            fixed (float* pInput = inputVec)
            {
                float actual = DequantQ4K.DotProduct(pData, pInput, elements);
                Assert.AreEqual(expected, actual, LooseTolerance,
                    "Q4_K multi-block DotProduct mismatch");
            }
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Write a float as FP16 (half precision) at the given offset in little-endian.
        /// </summary>
        private static void WriteHalf(byte[] data, int offset, float value)
        {
            ushort h = (ushort)BitConverter.HalfToUInt16Bits((Half)value);
            data[offset] = (byte)(h & 0xFF);
            data[offset + 1] = (byte)(h >> 8);
        }

        #endregion
    }
}
