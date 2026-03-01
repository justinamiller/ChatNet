using System;
using System.Buffers;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Tensors
{
    /// <summary>
    /// Manages pre-allocated and pooled tensor buffers for inference.
    /// Avoids per-token allocations by reusing buffers across forward passes.
    /// </summary>
    public sealed class TensorBuffer : IDisposable
    {
        private float[]? _buffer;
        private readonly int _size;

        public TensorBuffer(int size)
        {
            _size = size;
            _buffer = ArrayPool<float>.Shared.Rent(size);
        }

        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _size;
        }

        public Span<float> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _buffer.AsSpan(0, _size);
        }

        public float[] Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _buffer!;
        }

        /// <summary>Zero out the buffer.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Clear()
        {
            Span.Clear();
        }

        public void Dispose()
        {
            if (_buffer != null)
            {
                ArrayPool<float>.Shared.Return(_buffer);
                _buffer = null;
            }
        }
    }
}
