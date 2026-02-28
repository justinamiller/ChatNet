using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat;
using ChatNet.Core.Chat.Templates;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;
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

        /// <summary>Model configuration.</summary>
        public ModelConfig Config => _config;

        /// <summary>Tokenizer instance.</summary>
        public ITokenizer Tokenizer => _tokenizer;

        /// <summary>Model load time in milliseconds.</summary>
        public long LoadTimeMs { get; }

        private InferenceEngine(IModel model, ITokenizer tokenizer, MemoryMappedWeights weightLoader, ModelConfig config, long loadTimeMs)
        {
            _model = model;
            _tokenizer = tokenizer;
            _weightLoader = weightLoader;
            _config = config;
            LoadTimeMs = loadTimeMs;
        }

        /// <summary>
        /// Load a model from a GGUF file path.
        /// </summary>
        public static InferenceEngine LoadFromGguf(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {modelPath}");
            }

            var sw = Stopwatch.StartNew();

            // Step 1: Parse GGUF header and metadata
            var reader = new GgufReader(modelPath);
            reader.Load();

            // Step 2: Extract model config
            ModelConfig config = reader.ExtractModelConfig();

            // Step 3: Create memory-mapped weight access
            var weightLoader = new MemoryMappedWeights(modelPath, reader.Tensors, reader.TensorDataOffset);

            // Step 4: Create tokenizer from GGUF metadata
            var tokenizer = new BpeTokenizer(reader.Metadata);

            // Step 5: Create model (Llama family for now)
            var weights = new LlamaWeights(weightLoader, new LlamaConfig(config));
            var model = new LlamaModel(config, weights);

            sw.Stop();

            return new InferenceEngine(model, tokenizer, weightLoader, config, sw.ElapsedMilliseconds);
        }

        /// <summary>
        /// Generate text from a prompt, calling onToken for each generated token.
        /// Returns the total number of tokens generated.
        /// </summary>
        public int Generate(string prompt, ISampler sampler, int maxTokens, Action<string> onToken,
            string[]? stopStrings = null)
        {
            int vocabSize = _config.VocabSize;

            // Tokenize the prompt
            int[] tokenBuffer = ArrayPool<int>.Shared.Rent(_config.ContextLength);
            float[] logits = ArrayPool<float>.Shared.Rent(vocabSize);

            try
            {
                int promptTokenCount = _tokenizer.Encode(prompt.AsSpan(), tokenBuffer.AsSpan());

                // Forward pass for the entire prompt (prefill)
                ReadOnlySpan<int> promptTokens = tokenBuffer.AsSpan(0, promptTokenCount);

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

                // Track recent output for stop string detection using a char buffer
                char[] recentBuf = stopStrings != null ? new char[256] : Array.Empty<char>();
                int recentLen = 0;

                while (generatedCount < maxTokens)
                {
                    // Check for EOS
                    if (nextToken == _tokenizer.EosToken)
                    {
                        break;
                    }

                    // Decode and emit token
                    string tokenText = _tokenizer.Decode(nextToken);
                    onToken(tokenText);
                    generatedCount++;

                    // Check stop strings
                    if (stopStrings != null)
                    {
                        // Append to ring buffer without string allocation
                        for (int ci = 0; ci < tokenText.Length; ci++)
                        {
                            if (recentLen < recentBuf.Length)
                            {
                                recentBuf[recentLen++] = tokenText[ci];
                            }
                            else
                            {
                                // Shift left and append
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

                    // Check context length
                    if (currentPos >= _config.ContextLength - 1)
                    {
                        break;
                    }

                    // Forward pass for this token
                    ReadOnlySpan<int> singleToken = new ReadOnlySpan<int>(ref nextToken);
                    _model.Forward(singleToken, currentPos, logits.AsSpan(0, vocabSize));
                    currentPos++;

                    // Sample next token
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
        /// Generate a chat response using the TinyLlama chat template.
        /// </summary>
        public int GenerateChat(ChatSession session, ISampler sampler, int maxTokens, Action<string> onToken)
        {
            string prompt = session.BuildPrompt();
            return Generate(prompt, sampler, maxTokens, onToken, new[] { "</s>" });
        }

        public void Dispose()
        {
            _model.Dispose();
            _weightLoader.Dispose();
        }
    }
}
