namespace ChatNet.Core.Gguf
{
    /// <summary>
    /// Descriptor for a single tensor in a GGUF file.
    /// </summary>
    public sealed class GgufTensorInfo
    {
        public string Name { get; set; } = "";
        public int NDimensions { get; set; }
        public ulong[] Dimensions { get; set; } = System.Array.Empty<ulong>();
        public GgmlType Type { get; set; }
        public ulong Offset { get; set; }

        /// <summary>Total number of elements in this tensor.</summary>
        public ulong ElementCount
        {
            get
            {
                if (Dimensions.Length == 0) return 0;
                ulong count = 1;
                for (int i = 0; i < Dimensions.Length; i++)
                {
                    count *= Dimensions[i];
                }
                return count;
            }
        }

        /// <summary>Size in bytes of this tensor's data.</summary>
        public ulong ByteSize
        {
            get
            {
                ulong elements = ElementCount;
                switch (Type)
                {
                    case GgmlType.F32: return elements * 4;
                    case GgmlType.F16: return elements * 2;
                    case GgmlType.Q4_0:
                        // Q4_0: blocks of 32 elements, each block = 18 bytes (2 for scale + 16 for data)
                        ulong blocks = (elements + 31) / 32;
                        return blocks * 18;
                    case GgmlType.Q4_1:
                        blocks = (elements + 31) / 32;
                        return blocks * 20; // 2 scale + 2 min + 16 data
                    case GgmlType.Q8_0:
                        blocks = (elements + 31) / 32;
                        return blocks * 34; // 2 scale + 32 data
                    case GgmlType.Q6K:
                        // Q6_K: super-blocks of 256 elements
                        // ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
                        blocks = (elements + 255) / 256;
                        return blocks * 210;
                    case GgmlType.Q5_0:
                        blocks = (elements + 31) / 32;
                        return blocks * 22; // 2 scale + 4 high-bits + 16 data
                    case GgmlType.Q5_1:
                        blocks = (elements + 31) / 32;
                        return blocks * 24; // 2 scale + 2 min + 4 high-bits + 16 data
                    case GgmlType.Q2K:
                        blocks = (elements + 255) / 256;
                        return blocks * 84; // 256/16*2 scales + 256/4 quants + 2 d + 2 dmin
                    case GgmlType.Q3K:
                        blocks = (elements + 255) / 256;
                        return blocks * 110;
                    case GgmlType.Q4K:
                        blocks = (elements + 255) / 256;
                        return blocks * 144;
                    case GgmlType.Q5K:
                        blocks = (elements + 255) / 256;
                        return blocks * 176;
                    case GgmlType.Q8K:
                        blocks = (elements + 255) / 256;
                        return blocks * 292;
                    default:
                        return elements * 4; // fallback
                }
            }
        }
    }
}
