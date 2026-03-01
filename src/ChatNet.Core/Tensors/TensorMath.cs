using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using ChatNet.Core.Tensors.Quantization;

namespace ChatNet.Core.Tensors
{
    /// <summary>
    /// SIMD-accelerated tensor math operations for LLM inference.
    /// All hot-path operations use Vector256 with fallback to scalar.
    /// </summary>
    public static class TensorMath
    {
        /// <summary>
        /// Minimum output rows before using Parallel.For. Below this threshold,
        /// thread scheduling overhead exceeds the parallelism benefit.
        /// </summary>
        private const int ParallelRowThreshold = 512;
        /// <summary>
        /// RMS Norm: output[i] = input[i] * weight[i] / sqrt(mean(input^2) + eps)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, Span<float> output, float epsilon)
        {
            int n = input.Length;

            // Compute sum of squares using SIMD
            float sumSq = 0f;
            int i = 0;

            if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
            {
                var vSumSq = Vector<float>.Zero;
                int vecLen = Vector<float>.Count;
                int limit = n - vecLen + 1;
                for (; i < limit; i += vecLen)
                {
                    var v = new Vector<float>(input.Slice(i));
                    vSumSq += v * v;
                }
                sumSq = VectorSum(vSumSq);
            }

            for (; i < n; i++)
            {
                sumSq += input[i] * input[i];
            }

            float rms = 1.0f / MathF.Sqrt(sumSq / n + epsilon);

            // Apply norm: output[i] = input[i] * rms * weight[i]
            i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
            {
                var vRms = new Vector<float>(rms);
                int vecLen = Vector<float>.Count;
                int limit = n - vecLen + 1;
                for (; i < limit; i += vecLen)
                {
                    var vIn = new Vector<float>(input.Slice(i));
                    var vW = new Vector<float>(weight.Slice(i));
                    var result = vIn * vRms * vW;
                    result.CopyTo(output.Slice(i));
                }
            }

            for (; i < n; i++)
            {
                output[i] = input[i] * rms * weight[i];
            }
        }

        /// <summary>
        /// Softmax: output[i] = exp(input[i] - max) / sum(exp(input - max))
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Softmax(Span<float> data, int length)
        {
            // Find max
            float max = float.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                if (data[i] > max) max = data[i];
            }

            // Compute exp(x - max) and sum
            float sum = 0f;
            for (int i = 0; i < length; i++)
            {
                float val = MathF.Exp(data[i] - max);
                data[i] = val;
                sum += val;
            }

            // Normalize
            if (sum > 0f)
            {
                float invSum = 1.0f / sum;
                int i = 0;
                if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
                {
                    var vInvSum = new Vector<float>(invSum);
                    int vecLen = Vector<float>.Count;
                    int limit = length - vecLen + 1;
                    for (; i < limit; i += vecLen)
                    {
                        var v = new Vector<float>(data.Slice(i));
                        (v * vInvSum).CopyTo(data.Slice(i));
                    }
                }
                for (; i < length; i++)
                {
                    data[i] *= invSum;
                }
            }
        }

        /// <summary>
        /// SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        /// Applied element-wise: output[i] = gate[i] * silu_scalar(gate[i]) is wrong;
        /// Actually: output[i] = silu(gate[i]) * up[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SiluElementwiseMul(Span<float> gate, ReadOnlySpan<float> up, int length)
        {
            // gate[i] = silu(gate[i]) * up[i]
            for (int i = 0; i < length; i++)
            {
                float x = gate[i];
                float silu = x / (1.0f + MathF.Exp(-x));
                gate[i] = silu * up[i];
            }
        }

        /// <summary>
        /// Apply RoPE (Rotary Position Embeddings) to query and key vectors.
        /// Operates on pairs of dimensions.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ApplyRoPE(Span<float> q, Span<float> k, int position, int headDim, int nHeads, int nKvHeads, float freqBase)
        {
            // Apply RoPE to each head in q
            for (int h = 0; h < nHeads; h++)
            {
                int offset = h * headDim;
                ApplyRoPEToVector(q.Slice(offset, headDim), position, headDim, freqBase);
            }

            // Apply RoPE to each head in k
            for (int h = 0; h < nKvHeads; h++)
            {
                int offset = h * headDim;
                ApplyRoPEToVector(k.Slice(offset, headDim), position, headDim, freqBase);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ApplyRoPEToVector(Span<float> vec, int position, int headDim, float freqBase)
        {
            // RoPE: rotate pairs of dimensions by position-dependent angles
            for (int i = 0; i < headDim; i += 2)
            {
                float freq = 1.0f / MathF.Pow(freqBase, (float)i / headDim);
                float theta = position * freq;
                float cosTheta = MathF.Cos(theta);
                float sinTheta = MathF.Sin(theta);

                float x0 = vec[i];
                float x1 = vec[i + 1];
                vec[i] = x0 * cosTheta - x1 * sinTheta;
                vec[i + 1] = x0 * sinTheta + x1 * cosTheta;
            }
        }

        /// <summary>
        /// Matrix-vector multiply for float weights: output[i] = dot(weights[i,:], input)
        /// weights shape: [outDim, inDim], row-major.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MatVecMul(ReadOnlySpan<float> weights, ReadOnlySpan<float> input,
            Span<float> output, int outDim, int inDim)
        {
            fixed (float* pWeights = weights)
            fixed (float* pInput = input)
            fixed (float* pOutput = output)
            {
                for (int row = 0; row < outDim; row++)
                {
                    float* rowPtr = pWeights + (long)row * inDim;
                    pOutput[row] = DotProductSimd(rowPtr, pInput, inDim);
                }
            }
        }

        /// <summary>
        /// Matrix-vector multiply for Q4_0 quantized weights.
        /// weights: raw Q4_0 bytes for [outDim, inDim] matrix.
        /// Parallelizes across output rows when outDim is large enough.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatVecMulQ4_0(byte* weights, ReadOnlySpan<float> input,
            Span<float> output, int outDim, int inDim)
        {
            int bytesPerRow = (inDim / DequantQ4_0.BlockSize) * DequantQ4_0.BytesPerBlock;

            fixed (float* pInput = input)
            fixed (float* pOutput = output)
            {
                byte* w = weights;
                float* pIn = pInput;
                float* pOut = pOutput;

                if (outDim >= ParallelRowThreshold)
                {
                    int bpr = bytesPerRow;
                    int dim = inDim;
                    Parallel.For(0, outDim, row =>
                    {
                        byte* rowPtr = w + (long)row * bpr;
                        pOut[row] = DequantQ4_0.DotProduct(rowPtr, pIn, dim);
                    });
                }
                else
                {
                    for (int row = 0; row < outDim; row++)
                    {
                        byte* rowPtr = w + (long)row * bytesPerRow;
                        pOut[row] = DequantQ4_0.DotProduct(rowPtr, pIn, inDim);
                    }
                }
            }
        }

        /// <summary>
        /// Matrix-vector multiply for Q6_K quantized weights.
        /// weights: raw Q6_K bytes for [outDim, inDim] matrix.
        /// Parallelizes across output rows when outDim is large enough.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe void MatVecMulQ6K(byte* weights, ReadOnlySpan<float> input,
            Span<float> output, int outDim, int inDim)
        {
            int bytesPerRow = (inDim / DequantQ6K.BlockSize) * DequantQ6K.BytesPerBlock;

            fixed (float* pInput = input)
            fixed (float* pOutput = output)
            {
                byte* w = weights;
                float* pIn = pInput;
                float* pOut = pOutput;

                if (outDim >= ParallelRowThreshold)
                {
                    int bpr = bytesPerRow;
                    int dim = inDim;
                    Parallel.For(0, outDim, row =>
                    {
                        byte* rowPtr = w + (long)row * bpr;
                        pOut[row] = DequantQ6K.DotProduct(rowPtr, pIn, dim);
                    });
                }
                else
                {
                    for (int row = 0; row < outDim; row++)
                    {
                        byte* rowPtr = w + (long)row * bytesPerRow;
                        pOut[row] = DequantQ6K.DotProduct(rowPtr, pIn, inDim);
                    }
                }
            }
        }

        /// <summary>
        /// SIMD dot product of two float vectors.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProductSimd(float* a, float* b, int length)
        {
            float sum = 0f;
            int i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                int vecLen = Vector<float>.Count;
                var vSum = Vector<float>.Zero;
                int limit = length - vecLen + 1;
                for (; i < limit; i += vecLen)
                {
                    var va = *(Vector<float>*)(a + i);
                    var vb = *(Vector<float>*)(b + i);
                    vSum += va * vb;
                }
                sum = VectorSum(vSum);
            }

            for (; i < length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        /// <summary>
        /// Element-wise addition: output[i] = a[i] + b[i] (in-place on a).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(Span<float> a, ReadOnlySpan<float> b, int length)
        {
            int i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
            {
                int vecLen = Vector<float>.Count;
                int limit = length - vecLen + 1;
                for (; i < limit; i += vecLen)
                {
                    var va = new Vector<float>(a.Slice(i));
                    var vb = new Vector<float>(b.Slice(i));
                    (va + vb).CopyTo(a.Slice(i));
                }
            }
            for (; i < length; i++)
            {
                a[i] += b[i];
            }
        }

        /// <summary>
        /// Copy span contents.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Copy(ReadOnlySpan<float> src, Span<float> dst, int length)
        {
            src.Slice(0, length).CopyTo(dst.Slice(0, length));
        }

        /// <summary>
        /// Sum all elements in a SIMD vector.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float VectorSum(Vector<float> v)
        {
            float sum = 0f;
            for (int i = 0; i < Vector<float>.Count; i++)
            {
                sum += v[i];
            }
            return sum;
        }
    }
}
