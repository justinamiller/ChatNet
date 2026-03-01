namespace ChatNet.Core.Tensors.Quantization
{
    /// <summary>
    /// Quantization type enum matching GGML types.
    /// </summary>
    public enum QuantType
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
