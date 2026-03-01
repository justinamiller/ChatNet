using System;
using System.Buffers;
using System.Runtime.CompilerServices;

namespace ChatNet.Core.Memory
{
    /// <summary>
    /// ArrayPool wrapper with IDisposable for automatic return.
    /// Used for temporary buffers in inference.
    /// </summary>
    public struct PooledBuffer<T> : IDisposable
    {
        private T[]? _array;
        private readonly int _length;

        public PooledBuffer(int length)
        {
            _length = length;
            _array = ArrayPool<T>.Shared.Rent(length);
        }

        public readonly int Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _length;
        }

        public readonly Span<T> Span
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _array.AsSpan(0, _length);
        }

        public readonly T[] Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _array!;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Dispose()
        {
            if (_array != null)
            {
                ArrayPool<T>.Shared.Return(_array);
                _array = null;
            }
        }
    }
}
