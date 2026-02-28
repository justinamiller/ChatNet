using System;
using System.Runtime.CompilerServices;
using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Samplers
{
    /// <summary>
    /// Greedy (argmax) sampler: always picks the token with the highest logit.
    /// </summary>
    public sealed class GreedySampler : ISampler
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Sample(ReadOnlySpan<float> logits)
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
