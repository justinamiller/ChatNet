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
                for (int i = 0; i < vocabSize; i++)
                {
                    probs[i] = logits[i] / _temperature;
                    indices[i] = i;
                }

                // Softmax
                TensorMath.Softmax(probs.AsSpan(), vocabSize);

                // Sort by probability (descending) - simple insertion sort for top-k
                // We only need the top-k elements, so partial sort is fine
                int k = _topK < vocabSize ? _topK : vocabSize;
                for (int i = 0; i < k; i++)
                {
                    int maxIdx = i;
                    float maxVal = probs[i];
                    for (int j = i + 1; j < vocabSize; j++)
                    {
                        if (probs[j] > maxVal)
                        {
                            maxVal = probs[j];
                            maxIdx = j;
                        }
                    }
                    if (maxIdx != i)
                    {
                        // Swap probs
                        float tmpF = probs[i];
                        probs[i] = probs[maxIdx];
                        probs[maxIdx] = tmpF;
                        // Swap indices
                        int tmpI = indices[i];
                        indices[i] = indices[maxIdx];
                        indices[maxIdx] = tmpI;
                    }
                }

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
