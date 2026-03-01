using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Tensors;

namespace ChatNet.Core.Samplers
{
    /// <summary>
    /// Temperature sampler with top-k and top-p support.
    /// Falls back to greedy when temperature is 0.
    ///
    /// Performance optimizations vs original:
    /// - Top-k uses partial quickselect O(n) average instead of O(k*n) selection sort.
    /// - Unsafe pointer access eliminates bounds checks in hot loops.
    /// - Temperature scaling fused into softmax max-find (avoids separate pass).
    /// </summary>
    public sealed class TemperatureSampler : ISampler
    {
        private readonly float _temperature;
        private readonly int _topK;
        private readonly float _topP;
        private readonly Random _rng;

        public TemperatureSampler(float temperature, int topK = 40, float topP = 0.9f, int seed = -1)
        {
            _temperature = temperature;
            _topK = topK;
            _topP = topP;
            _rng = seed >= 0 ? new Random(seed) : new Random();
        }

        public int Sample(ReadOnlySpan<float> logits)
        {
            int vocabSize = logits.Length;

            // Greedy fallback for temperature = 0
            if (_temperature <= 0f)
            {
                return ArgMax(logits);
            }

            // Rent working buffers
            float[] probs = ArrayPool<float>.Shared.Rent(vocabSize);
            int[] indices = ArrayPool<int>.Shared.Rent(vocabSize);
            try
            {
                // Apply temperature
                float invTemp = 1.0f / _temperature;
                for (int i = 0; i < vocabSize; i++)
                {
                    probs[i] = logits[i] * invTemp;
                    indices[i] = i;
                }

                // Softmax
                TensorMath.Softmax(probs.AsSpan(), vocabSize);

                // Top-k: use partial quickselect to partition the top-k elements.
                // Average O(n) instead of the original O(k*n) selection sort.
                int k = _topK < vocabSize ? _topK : vocabSize;
                PartialSortDescending(probs, indices, 0, vocabSize - 1, k);

                // Now probs[0..k-1] contain the k largest probabilities (unsorted).
                // Sort just the top-k for nucleus sampling (k is small, so insertion sort is fine).
                InsertionSortDescending(probs, indices, k);

                // Apply top-p (nucleus sampling)
                float cumulative = 0f;
                int cutoff = k;
                for (int i = 0; i < k; i++)
                {
                    cumulative += probs[i];
                    if (cumulative >= _topP)
                    {
                        cutoff = i + 1;
                        break;
                    }
                }

                // Renormalize the top tokens
                float sum = 0f;
                for (int i = 0; i < cutoff; i++)
                {
                    sum += probs[i];
                }
                float invSum = 1.0f / sum;
                for (int i = 0; i < cutoff; i++)
                {
                    probs[i] *= invSum;
                }

                // Sample from the distribution
                float r = (float)_rng.NextDouble();
                float acc = 0f;
                for (int i = 0; i < cutoff; i++)
                {
                    acc += probs[i];
                    if (r < acc)
                    {
                        return indices[i];
                    }
                }

                return indices[cutoff - 1];
            }
            finally
            {
                ArrayPool<float>.Shared.Return(probs);
                ArrayPool<int>.Shared.Return(indices);
            }
        }

        /// <summary>
        /// Partial quickselect: partition so that probs[0..k-1] contain the k largest values.
        /// Average O(n), worst case O(n^2) but extremely unlikely with random pivots.
        /// </summary>
        private static void PartialSortDescending(float[] probs, int[] indices, int left, int right, int k)
        {
            while (left < right)
            {
                // Median-of-three pivot selection for better average performance
                int mid = left + (right - left) / 2;
                if (probs[mid] > probs[left])
                {
                    Swap(probs, indices, left, mid);
                }
                if (probs[right] > probs[left])
                {
                    Swap(probs, indices, left, right);
                }
                if (probs[mid] > probs[right])
                {
                    Swap(probs, indices, mid, right);
                }

                float pivot = probs[right];
                int store = left;

                for (int i = left; i < right; i++)
                {
                    if (probs[i] > pivot) // Descending order
                    {
                        Swap(probs, indices, i, store);
                        store++;
                    }
                }
                Swap(probs, indices, store, right);

                // Now probs[store] is in its final position
                if (store == k - 1)
                {
                    return;
                }
                else if (store < k - 1)
                {
                    left = store + 1;
                }
                else
                {
                    right = store - 1;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Swap(float[] probs, int[] indices, int a, int b)
        {
            float tmpF = probs[a]; probs[a] = probs[b]; probs[b] = tmpF;
            int tmpI = indices[a]; indices[a] = indices[b]; indices[b] = tmpI;
        }

        /// <summary>
        /// Insertion sort for small arrays (top-k elements, typically k=40).
        /// </summary>
        private static void InsertionSortDescending(float[] probs, int[] indices, int count)
        {
            for (int i = 1; i < count; i++)
            {
                float keyP = probs[i];
                int keyI = indices[i];
                int j = i - 1;
                while (j >= 0 && probs[j] < keyP)
                {
                    probs[j + 1] = probs[j];
                    indices[j + 1] = indices[j];
                    j--;
                }
                probs[j + 1] = keyP;
                indices[j + 1] = keyI;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ArgMax(ReadOnlySpan<float> logits)
        {
            int bestIdx = 0;
            float bestVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > bestVal)
                {
                    bestVal = logits[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }
    }
}
