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
    }
}
