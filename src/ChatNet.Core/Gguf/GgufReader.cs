using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Gguf
{
    /// <summary>
    /// Parses a GGUF file: header, metadata KVs, tensor descriptors, and computes data offsets.
    /// Uses buffered FileStream reads with explicit little-endian decoding.
    /// </summary>
    public sealed class GgufReader
    {
        private readonly string _filePath;
        private readonly byte[] _readBuffer;

        public GgufMetadata Metadata { get; private set; } = new GgufMetadata();
        public GgufTensorInfo[] Tensors { get; private set; } = Array.Empty<GgufTensorInfo>();
        public ulong TensorDataOffset { get; private set; }
        public int Alignment { get; private set; } = GgufConstants.DefaultAlignment;

        public GgufReader(string filePath)
        {
            _filePath = filePath;
            _readBuffer = new byte[65536]; // 64KB read buffer
        }

        /// <summary>
        /// Parse the entire GGUF file header, metadata, and tensor info.
        /// </summary>
        public void Load()
        {
            using var stream = new FileStream(_filePath, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);

            // Read magic
            uint magic = ReadUInt32(stream);
            if (magic != GgufConstants.Magic)
            {
                throw new InvalidDataException($"Invalid GGUF magic: 0x{magic:X8}, expected 0x{GgufConstants.Magic:X8}");
            }

            // Read version
            uint version = ReadUInt32(stream);
            if (version < 2 || version > 3)
            {
                throw new InvalidDataException($"Unsupported GGUF version: {version}");
            }

            // Read counts
            ulong tensorCount = ReadUInt64(stream);
            ulong metadataKvCount = ReadUInt64(stream);

            // Parse metadata key-value pairs
            for (ulong i = 0; i < metadataKvCount; i++)
            {
                ReadMetadataKV(stream);
            }

            // Check for custom alignment in metadata
            Alignment = Metadata.GetInt32("general.alignment", GgufConstants.DefaultAlignment);

            // Parse tensor info descriptors
            Tensors = new GgufTensorInfo[tensorCount];
            for (ulong i = 0; i < tensorCount; i++)
            {
                Tensors[i] = ReadTensorInfo(stream);
            }

            // Tensor data starts after all headers, aligned
            long currentPosition = stream.Position;
            TensorDataOffset = AlignOffset((ulong)currentPosition, Alignment);
        }

        /// <summary>
        /// Extract a ModelConfig from parsed metadata.
        /// </summary>
        public ModelConfig ExtractModelConfig()
        {
            string arch = Metadata.GetString("general.architecture", "llama");

            // Helper to read int with architecture-specific key and fallback aliases
            int GetArchInt(string suffix, int defaultValue)
            {
                int val = Metadata.GetInt32(arch + "." + suffix, -1);
                if (val >= 0) return val;
                return Metadata.GetInt32(suffix, defaultValue);
            }

            // Helper to read float with architecture-specific key, then bare key fallback.
            // Uses TryGet to handle the case where the value is stored as int/uint.
            float GetArchFloat(string suffix, float defaultValue)
            {
                string archKey = arch + "." + suffix;
                if (Metadata.TryGet<float>(archKey, out float fv)) return fv;
                if (Metadata.TryGet<double>(archKey, out double dv)) return (float)dv;
                // Some quantizers store freq_base as int
                if (Metadata.TryGet<uint>(archKey, out uint uv)) return uv;
                if (Metadata.TryGet<int>(archKey, out int iv)) return iv;

                // Fallback to bare key
                if (Metadata.TryGet<float>(suffix, out fv)) return fv;
                if (Metadata.TryGet<double>(suffix, out dv)) return (float)dv;
                if (Metadata.TryGet<uint>(suffix, out uv)) return uv;
                if (Metadata.TryGet<int>(suffix, out iv)) return iv;

                return defaultValue;
            }

            int nHeads = GetArchInt("attention.head_count", 32);
            // Default KV heads to n_heads (full MHA) if not specified.
            // GQA models always store head_count_kv explicitly; omission implies MHA.
            int nKvHeads = GetArchInt("attention.head_count_kv", nHeads);

            var config = new ModelConfig
            {
                Architecture = arch,
                ModelName = Metadata.GetString("general.name", "Unknown"),
                EmbeddingDim = GetArchInt("embedding_length", 2048),
                LayerCount = GetArchInt("block_count", 22),
                AttentionHeadCount = nHeads,
                KeyValueHeadCount = nKvHeads,
                FeedForwardDim = GetArchInt("feed_forward_length", 5632),
                ContextLength = GetArchInt("context_length", 2048),
                RopeFreqBase = GetArchFloat("rope.freq_base", 10000.0f),
                RmsNormEpsilon = GetArchFloat("attention.layer_norm_rms_epsilon", 1e-5f),
                BosTokenId = Metadata.GetInt32("tokenizer.ggml.bos_token_id", 1),
                EosTokenId = Metadata.GetInt32("tokenizer.ggml.eos_token_id", 2),
                // Gemma 2 soft-capping
                AttnLogitSoftcap = GetArchFloat("attn_logit_softcapping", 0f),
                FinalLogitSoftcap = GetArchFloat("final_logit_softcapping", 0f),
            };

            // HeadDim: prefer explicit GGUF key (Gemma 2 has head_dim != dim/n_heads)
            int explicitHeadDim = GetArchInt("attention.key_length", -1);
            config.HeadDim = explicitHeadDim > 0
                ? explicitHeadDim
                : config.EmbeddingDim / config.AttentionHeadCount;

            // RotaryDim: partial RoPE (Phi-3 uses half of headDim)
            // 1. Check rope.partial_rotary_factor (float, e.g. 0.5)
            // 2. Check rope.dimension_count if it's less than headDim
            // 3. Phi-3 architecture fallback: always partial_rotary_factor=0.5
            int ropeDimCount = GetArchInt("rope.dimension_count", -1);
            float partialFactor = GetArchFloat("rope.partial_rotary_factor", 0f);

            if (partialFactor > 0f && partialFactor < 1f)
            {
                // Explicit partial rotary factor
                config.RotaryDim = (int)(config.HeadDim * partialFactor);
            }
            else if (ropeDimCount > 0 && ropeDimCount < config.HeadDim)
            {
                // Explicit rotary dimension less than headDim
                config.RotaryDim = ropeDimCount;
            }
            else if (arch == "phi3")
            {
                // Phi-3 always uses partial_rotary_factor=0.5 but some GGUF producers
                // store rope.dimension_count = headDim instead of the partial value
                config.RotaryDim = config.HeadDim / 2;
            }
            else
            {
                config.RotaryDim = config.HeadDim;
            }

            // Vocab size from token array length if available, else from metadata
            string[]? tokens = Metadata.GetStringArray("tokenizer.ggml.tokens");
            if (tokens != null)
            {
                config.VocabSize = tokens.Length;
            }
            else
            {
                config.VocabSize = GetArchInt("vocab_size", 32000);
            }

            return config;
        }

        private void ReadMetadataKV(Stream stream)
        {
            string key = ReadString(stream);
            GgufMetadataValueType valueType = (GgufMetadataValueType)ReadUInt32(stream);
            object value = ReadMetadataValue(stream, valueType);
            Metadata.Set(key, value);
        }

        private object ReadMetadataValue(Stream stream, GgufMetadataValueType valueType)
        {
            switch (valueType)
            {
                case GgufMetadataValueType.Uint8: return ReadByte(stream);
                case GgufMetadataValueType.Int8: return (sbyte)ReadByte(stream);
                case GgufMetadataValueType.Uint16: return ReadUInt16(stream);
                case GgufMetadataValueType.Int16: return (short)ReadUInt16(stream);
                case GgufMetadataValueType.Uint32: return ReadUInt32(stream);
                case GgufMetadataValueType.Int32: return (int)ReadUInt32(stream);
                case GgufMetadataValueType.Float32: return ReadFloat32(stream);
                case GgufMetadataValueType.Bool: return ReadByte(stream) != 0;
                case GgufMetadataValueType.String: return ReadString(stream);
                case GgufMetadataValueType.Uint64: return ReadUInt64(stream);
                case GgufMetadataValueType.Int64: return (long)ReadUInt64(stream);
                case GgufMetadataValueType.Float64: return ReadFloat64(stream);
                case GgufMetadataValueType.Array: return ReadArray(stream);
                default:
                    throw new InvalidDataException($"Unknown GGUF metadata value type: {valueType}");
            }
        }

        private object ReadArray(Stream stream)
        {
            GgufMetadataValueType elementType = (GgufMetadataValueType)ReadUInt32(stream);
            ulong length = ReadUInt64(stream);

            switch (elementType)
            {
                case GgufMetadataValueType.String:
                {
                    string[] arr = new string[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = ReadString(stream);
                    }
                    return arr;
                }
                case GgufMetadataValueType.Float32:
                {
                    float[] arr = new float[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = ReadFloat32(stream);
                    }
                    return arr;
                }
                case GgufMetadataValueType.Int32:
                {
                    int[] arr = new int[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = (int)ReadUInt32(stream);
                    }
                    return arr;
                }
                case GgufMetadataValueType.Uint32:
                {
                    int[] arr = new int[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = (int)ReadUInt32(stream);
                    }
                    return arr;
                }
                case GgufMetadataValueType.Uint16:
                {
                    int[] arr = new int[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = ReadUInt16(stream);
                    }
                    return arr;
                }
                case GgufMetadataValueType.Int8:
                {
                    int[] arr = new int[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = (sbyte)ReadByte(stream);
                    }
                    return arr;
                }
                case GgufMetadataValueType.Uint8:
                {
                    int[] arr = new int[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = ReadByte(stream);
                    }
                    return arr;
                }
                default:
                {
                    // Generic fallback: read as objects
                    object[] arr = new object[length];
                    for (ulong i = 0; i < length; i++)
                    {
                        arr[i] = ReadMetadataValue(stream, elementType);
                    }
                    return arr;
                }
            }
        }

        private GgufTensorInfo ReadTensorInfo(Stream stream)
        {
            var info = new GgufTensorInfo();
            info.Name = ReadString(stream);
            uint nDims = ReadUInt32(stream);
            info.NDimensions = (int)nDims;
            info.Dimensions = new ulong[nDims];
            for (int i = 0; i < (int)nDims; i++)
            {
                info.Dimensions[i] = ReadUInt64(stream);
            }
            info.Type = (GgmlType)ReadUInt32(stream);
            info.Offset = ReadUInt64(stream);
            return info;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong AlignOffset(ulong offset, int alignment)
        {
            ulong mask = (ulong)(alignment - 1);
            return (offset + mask) & ~mask;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private byte ReadByte(Stream stream)
        {
            int b = stream.ReadByte();
            if (b < 0) throw new EndOfStreamException();
            return (byte)b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ushort ReadUInt16(Stream stream)
        {
            ReadExact(stream, _readBuffer, 2);
            return BinaryPrimitives.ReadUInt16LittleEndian(_readBuffer.AsSpan(0, 2));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private uint ReadUInt32(Stream stream)
        {
            ReadExact(stream, _readBuffer, 4);
            return BinaryPrimitives.ReadUInt32LittleEndian(_readBuffer.AsSpan(0, 4));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ulong ReadUInt64(Stream stream)
        {
            ReadExact(stream, _readBuffer, 8);
            return BinaryPrimitives.ReadUInt64LittleEndian(_readBuffer.AsSpan(0, 8));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ReadFloat32(Stream stream)
        {
            ReadExact(stream, _readBuffer, 4);
            return BitConverter.Int32BitsToSingle(
                BinaryPrimitives.ReadInt32LittleEndian(_readBuffer.AsSpan(0, 4)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double ReadFloat64(Stream stream)
        {
            ReadExact(stream, _readBuffer, 8);
            return BitConverter.Int64BitsToDouble(
                BinaryPrimitives.ReadInt64LittleEndian(_readBuffer.AsSpan(0, 8)));
        }

        private string ReadString(Stream stream)
        {
            ulong length = ReadUInt64(stream);
            if (length == 0) return "";
            if (length > 1_000_000)
            {
                throw new InvalidDataException($"GGUF string length too large: {length}");
            }

            int len = (int)length;
            byte[] buf = len <= _readBuffer.Length ? _readBuffer : new byte[len];
            ReadExact(stream, buf, len);
            return Encoding.UTF8.GetString(buf, 0, len);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ReadExact(Stream stream, byte[] buffer, int count)
        {
            int offset = 0;
            while (offset < count)
            {
                int read = stream.Read(buffer, offset, count - offset);
                if (read == 0) throw new EndOfStreamException();
                offset += read;
            }
        }
    }
}
