namespace ChatNet.Core.Tokenizer
{
    /// <summary>
    /// Token type classifications from GGUF metadata.
    /// </summary>
    public enum TokenType
    {
        Normal = 1,
        Unknown = 2,
        Control = 3,
        UserDefined = 4,
        Unused = 5,
        Byte = 6,
    }
}
