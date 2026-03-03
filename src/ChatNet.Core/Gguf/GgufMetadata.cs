using System;
using System.Collections.Generic;

namespace ChatNet.Core.Gguf
{
    /// <summary>
    /// Stores metadata key-value pairs parsed from GGUF header.
    /// Uses Dictionary at load time (not hot path).
    /// </summary>
    public sealed class GgufMetadata
    {
        private readonly Dictionary<string, object> _values = new Dictionary<string, object>(256);

        public void Set(string key, object value)
        {
            _values[key] = value;
        }

        public bool TryGet<T>(string key, out T value)
        {
            if (_values.TryGetValue(key, out object? obj) && obj is T typed)
            {
                value = typed;
                return true;
            }
            value = default!;
            return false;
        }

        public T Get<T>(string key, T defaultValue)
        {
            if (_values.TryGetValue(key, out object? obj) && obj is T typed)
            {
                return typed;
            }
            return defaultValue;
        }

        public T GetRequired<T>(string key)
        {
            if (_values.TryGetValue(key, out object? obj) && obj is T typed)
            {
                return typed;
            }
            throw new InvalidOperationException($"Required GGUF metadata key not found or wrong type: '{key}' (expected {typeof(T).Name})");
        }

        public string GetString(string key, string defaultValue = "")
        {
            return Get(key, defaultValue);
        }

        public int GetInt32(string key, int defaultValue = 0)
        {
            if (_values.TryGetValue(key, out object? obj))
            {
                if (obj is int i) return i;
                if (obj is uint u) return (int)u;
                if (obj is long l) return (int)l;
                if (obj is ulong ul) return (int)ul;
            }
            return defaultValue;
        }

        public uint GetUInt32(string key, uint defaultValue = 0)
        {
            if (_values.TryGetValue(key, out object? obj))
            {
                if (obj is uint u) return u;
                if (obj is int i) return (uint)i;
                if (obj is long l) return (uint)l;
                if (obj is ulong ul) return (uint)ul;
            }
            return defaultValue;
        }

        public float GetFloat32(string key, float defaultValue = 0f)
        {
            if (_values.TryGetValue(key, out object? obj))
            {
                if (obj is float f) return f;
                if (obj is double d) return (float)d;
                if (obj is uint u) return u;
                if (obj is int i) return i;
                if (obj is ulong ul) return ul;
                if (obj is long l) return l;
            }
            return defaultValue;
        }

        public string[]? GetStringArray(string key)
        {
            if (_values.TryGetValue(key, out object? obj) && obj is string[] arr)
            {
                return arr;
            }
            return null;
        }

        public float[]? GetFloatArray(string key)
        {
            if (_values.TryGetValue(key, out object? obj) && obj is float[] arr)
            {
                return arr;
            }
            return null;
        }

        public int[]? GetIntArray(string key)
        {
            if (_values.TryGetValue(key, out object? obj) && obj is int[] arr)
            {
                return arr;
            }
            return null;
        }

        public IEnumerable<string> Keys => _values.Keys;
    }
}
