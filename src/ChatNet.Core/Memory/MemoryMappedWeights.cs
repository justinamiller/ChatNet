using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Gguf;

namespace ChatNet.Core.Memory
{
    /// <summary>
    /// Memory-mapped file access for model weights. Zero-copy tensor access.
    /// </summary>
    public sealed class MemoryMappedWeights : IWeightLoader
    {
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly Dictionary<string, GgufTensorInfo> _tensorMap;
        private readonly ulong _dataOffset;
        private readonly unsafe byte* _basePointer;
        private bool _disposed;

        public unsafe MemoryMappedWeights(string filePath, GgufTensorInfo[] tensors, ulong dataOffset)
        {
            _dataOffset = dataOffset;
            _tensorMap = new Dictionary<string, GgufTensorInfo>(tensors.Length);
            for (int i = 0; i < tensors.Length; i++)
            {
                _tensorMap[tensors[i].Name] = tensors[i];
            }

            _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

            // Get the base pointer for direct memory access
            byte* ptr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            _basePointer = ptr;
        }

        public bool HasTensor(string tensorName)
        {
            return _tensorMap.ContainsKey(tensorName);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe ReadOnlySpan<byte> GetTensorData(string tensorName)
        {
            if (!_tensorMap.TryGetValue(tensorName, out GgufTensorInfo? info))
            {
                throw new KeyNotFoundException($"Tensor not found: {tensorName}");
            }

            ulong absoluteOffset = _dataOffset + info.Offset;
            ulong byteSize = info.ByteSize;
            return new ReadOnlySpan<byte>(_basePointer + absoluteOffset, (int)byteSize);
        }

        /// <summary>
        /// Get tensor info by name.
        /// </summary>
        public GgufTensorInfo GetTensorInfo(string tensorName)
        {
            if (!_tensorMap.TryGetValue(tensorName, out GgufTensorInfo? info))
            {
                throw new KeyNotFoundException($"Tensor not found: {tensorName}");
            }
            return info;
        }

        /// <summary>
        /// Get a pointer to tensor data for unsafe direct access.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe byte* GetTensorPointer(string tensorName)
        {
            if (!_tensorMap.TryGetValue(tensorName, out GgufTensorInfo? info))
            {
                throw new KeyNotFoundException($"Tensor not found: {tensorName}");
            }
            ulong absoluteOffset = _dataOffset + info.Offset;
            return _basePointer + absoluteOffset;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _accessor.Dispose();
                _mmf.Dispose();
            }
        }
    }
}
