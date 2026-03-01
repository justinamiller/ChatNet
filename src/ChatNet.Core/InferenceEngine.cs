using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat;
using ChatNet.Core.Chat.Templates;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;
using ChatNet.Core.Models;
using ChatNet.Core.Models.Llama;
using ChatNet.Core.Samplers;
using ChatNet.Core.Tokenizer;

namespace ChatNet.Core
{
    /// <summary>
    /// Top-level orchestrator: load model -> tokenize -> infer -> sample -> decode -> repeat.
    /// </summary>
    public sealed class InferenceEngine : IDisposable
    {
        private readonly IModel _model;
        private readonly ITokenizer _tokenizer;
        private readonly MemoryMappedWeights _weightLoader;
        private readonly ModelConfig _config;
        private readonly ModelType _modelType;

        /// <summary>Model configuration.</summary>
        public ModelConfig Config => _config;

        /// <summary>Tokenizer instance.</summary>
        public ITokenizer Tokenizer => _tokenizer;

        /// <summary>Detected model architecture type.</summary>
        public ModelType Architecture => _modelType;

        /// <summary>Model load time in milliseconds.</summary>
        public long LoadTimeMs { get; }

        /// <summary>Enable debug diagnostics to stderr.</summary>
        public static bool DebugEnabled { get; set; }

        private InferenceEngine(IModel model, ITokenizer tokenizer, MemoryMappedWeights weightLoader,
            ModelConfig config, ModelType modelType, long loadTimeMs)
        {
            _model = model;
            _tokenizer = tokenizer;
            _weightLoader = weightLoader;
            _config = config;
            _modelType = modelType;
            LoadTimeMs = loadTimeMs;
        }

        /// <summary>
        /// Load a model from a GGUF file path.
        /// </summary>
        public static InferenceEngine LoadFromGguf(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException("Model file not found: " + modelPath);
            }

            var sw = Stopwatch.StartNew();

            // Step 1: Parse GGUF header and metadata
            var reader = new GgufReader(modelPath);
            reader.Load();

            if (DebugEnabled)
            {
                Console.Error.WriteLine("[DEBUG] GGUF parsed: " + reader.Tensors.Length +
                    " tensors, dataOffset=0x" + reader.TensorDataOffset.ToString("X") +
                    " alignment=" + reader.Alignment);
            }

            // Step 2: Extract model config and detect architecture
            ModelConfig config = reader.ExtractModelConfig();
            ModelType modelType = ModelFactory.DetectArchitecture(config.Architecture);

            if (DebugEnabled)
            {
                Console.Error.WriteLine("[DEBUG] Detected architecture: " + modelType +
                    " (from '" + config.Architecture + "')");
            }

            // Step 3: Create memory-mapped weight access
            var weightLoader = new MemoryMappedWeights(modelPath, reader.Tensors, reader.TensorDataOffset);

            // Step 4: Create tokenizer from GGUF metadata
            var tokenizer = new BpeTokenizer(reader.Metadata);

            if (DebugEnabled)
            {
                Console.Error.WriteLine("[DEBUG] Tokenizer: vocabSize=" + tokenizer.VocabSize +
                    " bos=" + tokenizer.BosToken + " eos=" + tokenizer.EosToken);
            }

            // Step 5: Create model via factory routing
            IModel model = ModelFactory.CreateModel(modelType, config, weightLoader, DebugEnabled);

            // Step 6: Emit tensor diagnostic summary
            if (DebugEnabled)
            {
                EmitTensorDiagnostics(reader.Tensors, config, modelType);
            }

            sw.Stop();

            return new InferenceEngine(model, tokenizer, weightLoader, config, modelType, sw.ElapsedMilliseconds);
        }

        private static void EmitTensorDiagnostics(GgufTensorInfo[] tensors, ModelConfig config, ModelType modelType)
        {
            // Build a quick name set for O(1) lookup
            var tensorNames = new System.Collections.Generic.HashSet<string>(tensors.Length);
            for (int i = 0; i < tensors.Length; i++)
            {
                tensorNames.Add(tensors[i].Name);
            }

            int mapped = 0;
            int missing = 0;

            // Check critical tensors
            string[] criticalGlobal = new[] { "token_embd.weight", "output_norm.weight" };
            for (int i = 0; i < criticalGlobal.Length; i++)
            {
                if (tensorNames.Contains(criticalGlobal[i]))
                    mapped++;
                else
                    missing++;
            }

            // output.weight is optional (may be tied to embeddings)
            if (tensorNames.Contains("output.weight"))
                mapped++;

            // Check per-layer tensors
            string[] perLayerSuffixes = new[]
            {
                ".attn_norm.weight", ".attn_q.weight", ".attn_k.weight",
                ".attn_v.weight", ".attn_output.weight",
                ".ffn_norm.weight", ".ffn_up.weight", ".ffn_down.weight"
            };

            for (int l = 0; l < config.LayerCount; l++)
            {
                string prefix = "blk." + l.ToString();
                for (int s = 0; s < perLayerSuffixes.Length; s++)
                {
                    if (tensorNames.Contains(prefix + perLayerSuffixes[s]))
                        mapped++;
                    else
                        missing++;
                }

                // ffn_gate is optional (not all architectures have it)
                if (tensorNames.Contains(prefix + ".ffn_gate.weight"))
                    mapped++;
            }

            Console.Error.WriteLine("[DEBUG] Tensor mapping for " + modelType + ": " +
                mapped + " mapped, " + missing + " missing critical");
        }

        /// <summary>
        /// Generate text from a prompt, calling onToken for each generated token.
        /// Returns the total number of tokens generated.
        /// </summary>
        public int Generate(string prompt, ISampler sampler, int maxTokens, Action<string> onToken,
            string[]? stopStrings = null)
        {
            int vocabSize = _config.VocabSize;

            int[] tokenBuffer = ArrayPool<int>.Shared.Rent(_config.ContextLength);
            float[] logits = ArrayPool<float>.Shared.Rent(vocabSize);

            try
            {
                int promptTokenCount = _tokenizer.Encode(prompt.AsSpan(), tokenBuffer.AsSpan());

                if (DebugEnabled)
                {
                    Console.Error.Write("[DEBUG] Encoded " + promptTokenCount + " tokens: [");
                    int printCount = promptTokenCount < 30 ? promptTokenCount : 30;
                    for (int di = 0; di < printCount; di++)
                    {
                        if (di > 0) Console.Error.Write(",");
                        Console.Error.Write(tokenBuffer[di]);
                    }
                    if (promptTokenCount > 30) Console.Error.Write("...");
                    Console.Error.WriteLine("]");

                    Console.Error.Write("[DEBUG] Decoded: ");
                    for (int di = 0; di < printCount; di++)
                    {
                        string decoded = _tokenizer.Decode(tokenBuffer[di]);
                        Console.Error.Write("'" + decoded + "'");
                    }
                    Console.Error.WriteLine();
                }

                // Process prompt tokens one at a time (sequential for KV cache)
                for (int i = 0; i < promptTokenCount; i++)
                {
                    ReadOnlySpan<int> token = tokenBuffer.AsSpan(i, 1);
                    _model.Forward(token, i, logits.AsSpan(0, vocabSize));
                }

                // Sample first generated token
                int nextToken = sampler.Sample(logits.AsSpan(0, vocabSize));
                int generatedCount = 0;
                int currentPos = promptTokenCount;

                if (DebugEnabled)
                {
                    Console.Error.WriteLine("[DEBUG] First sampled token: " + nextToken +
                        " ('" + _tokenizer.Decode(nextToken) + "')");
                }

                // Track recent output for stop string detection using a char buffer
                char[] recentBuf = stopStrings != null ? new char[256] : Array.Empty<char>();
                int recentLen = 0;

                while (generatedCount < maxTokens)
                {
                    if (nextToken == _tokenizer.EosToken)
                    {
                        break;
                    }

                    string tokenText = _tokenizer.Decode(nextToken);
                    onToken(tokenText);
                    generatedCount++;

                    if (stopStrings != null)
                    {
                        for (int ci = 0; ci < tokenText.Length; ci++)
                        {
                            if (recentLen < recentBuf.Length)
                            {
                                recentBuf[recentLen++] = tokenText[ci];
                            }
                            else
                            {
                                Array.Copy(recentBuf, 1, recentBuf, 0, recentBuf.Length - 1);
                                recentBuf[recentBuf.Length - 1] = tokenText[ci];
                            }
                        }

                        bool shouldStop = false;
                        ReadOnlySpan<char> recentSpan = recentBuf.AsSpan(0, recentLen);
                        for (int s = 0; s < stopStrings.Length; s++)
                        {
                            if (recentSpan.IndexOf(stopStrings[s].AsSpan()) >= 0)
                            {
                                shouldStop = true;
                                break;
                            }
                        }
                        if (shouldStop) break;
                    }

                    if (currentPos >= _config.ContextLength - 1)
                    {
                        break;
                    }

                    // Forward pass for this token - reuse tokenBuffer[0] to avoid
                    // reliance on ReadOnlySpan<int>(ref local) constructor
                    tokenBuffer[0] = nextToken;
                    _model.Forward(tokenBuffer.AsSpan(0, 1), currentPos, logits.AsSpan(0, vocabSize));
                    currentPos++;

                    nextToken = sampler.Sample(logits.AsSpan(0, vocabSize));
                }

                return generatedCount;
            }
            finally
            {
                ArrayPool<int>.Shared.Return(tokenBuffer);
                ArrayPool<float>.Shared.Return(logits);
            }
        }

        /// <summary>
        /// Generate a chat response using the architecture-appropriate chat template.
        /// </summary>
        public int GenerateChat(ChatSession session, ISampler sampler, int maxTokens, Action<string> onToken)
        {
            string prompt = session.BuildPrompt();

            if (DebugEnabled)
            {
                Console.Error.WriteLine("[DEBUG] Chat prompt:");
                Console.Error.WriteLine(prompt);
                Console.Error.WriteLine("[DEBUG] --- end prompt ---");
            }

            string[] stopStrings = ModelFactory.GetStopStrings(_modelType);
            return Generate(prompt, sampler, maxTokens, onToken, stopStrings);
        }

        public void Dispose()
        {
            _model.Dispose();
            _weightLoader.Dispose();
        }
    }
}
