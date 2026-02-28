namespace ChatNet.Core.Gguf
{
    /// <summary>
    /// Constants for the GGUF file format.
    /// </summary>
    internal static class GgufConstants
    {
        /// <summary>GGUF magic number: "GGUF" in little-endian.</summary>
        public const uint Magic = 0x46475547;

        /// <summary>Expected GGUF version.</summary>
        public const uint ExpectedVersion = 3;

        /// <summary>Default alignment for tensor data.</summary>
        public const int DefaultAlignment = 32;
    }

    /// <summary>
    /// GGUF metadata value types.
    /// </summary>
    internal enum GgufMetadataValueType : uint
    {
        Uint8 = 0,
        Int8 = 1,
        Uint16 = 2,
        Int16 = 3,
        Uint32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
        Uint64 = 10,
        Int64 = 11,
        Float64 = 12,
    }

    /// <summary>
    /// GGML tensor data types.
    /// </summary>
    public enum GgmlType : uint
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2K = 10,
        Q3K = 11,
        Q4K = 12,
        Q5K = 13,
        Q6K = 14,
        Q8K = 15,
        IQ2XXS = 16,
        IQ2XS = 17,
        IQ3XXS = 18,
        IQ1S = 19,
        IQ4NL = 20,
        IQ3S = 21,
        IQ2S = 22,
        IQ4XS = 23,
        I8 = 24,
        I16 = 25,
        I32 = 26,
        I64 = 27,
        F64 = 28,
        IQ1M = 29,
    }
}
