using System;

namespace ChatNet.Core.Abstractions
{
    /// <summary>
    /// Model weight loading abstraction, allowing different model formats.
    /// </summary>
    public interface IWeightLoader : IDisposable
    {
        /// <summary>Get a span of bytes for a named tensor's raw data.</summary>
        ReadOnlySpan<byte> GetTensorData(string tensorName);

        /// <summary>Check if a tensor exists by name.</summary>
        bool HasTensor(string tensorName);
    }
}
