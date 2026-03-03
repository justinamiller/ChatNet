using System;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat.Templates;
using ChatNet.Core.Gguf;
using ChatNet.Core.Memory;
using ChatNet.Core.Models.Gemma;
using ChatNet.Core.Models.Llama;
using ChatNet.Core.Models.Mistral;
using ChatNet.Core.Models.Phi;
using ChatNet.Core.Models.Qwen;

namespace ChatNet.Core.Models
{
    /// <summary>
    /// Factory that detects model architecture from GGUF metadata
    /// and creates the appropriate IModel and IChatTemplate instances.
    /// </summary>
    public static class ModelFactory
    {
        /// <summary>
        /// Detect the ModelType from GGUF metadata general.architecture key.
        /// </summary>
        public static ModelType DetectArchitecture(string architecture)
        {
            string arch = architecture.ToLowerInvariant();

            if (arch == "llama")
                return ModelType.Llama;
            if (arch == "qwen2" || arch == "qwen")
                return ModelType.Qwen;
            if (arch == "mistral")
                return ModelType.Mistral;
            if (arch == "gemma" || arch == "gemma2")
                return ModelType.Gemma;
            if (arch == "phi2" || arch == "phi3" || arch == "phi")
                return ModelType.Phi;

            Console.Error.WriteLine("[WARN] Unknown architecture '" + architecture + "', defaulting to Llama.");
            return ModelType.Llama;
        }

        /// <summary>
        /// Create the appropriate IModel for the detected architecture.
        /// </summary>
        public static IModel CreateModel(ModelType modelType, ModelConfig config,
            MemoryMappedWeights weightLoader, bool debugEnabled)
        {
            switch (modelType)
            {
                case ModelType.Llama:
                {
                    LlamaModel.DebugEnabled = debugEnabled;
                    var llamaCfg = new LlamaConfig(config);
                    var weights = new LlamaWeights(weightLoader, llamaCfg);
                    return new LlamaModel(config, weights);
                }

                case ModelType.Qwen:
                {
                    QwenModel.DebugEnabled = debugEnabled;
                    var qwenCfg = new QwenConfig(config);
                    var weights = new QwenWeights(weightLoader, qwenCfg);
                    return new QwenModel(config, weights);
                }

                case ModelType.Mistral:
                {
                    MistralModel.DebugEnabled = debugEnabled;
                    var mistralCfg = new MistralConfig(config);
                    var weights = new MistralWeights(weightLoader, mistralCfg);
                    return new MistralModel(config, weights);
                }

                case ModelType.Gemma:
                {
                    GemmaModel.DebugEnabled = debugEnabled;
                    var gemmaCfg = new GemmaConfig(config);
                    var weights = new GemmaWeights(weightLoader, gemmaCfg);
                    return new GemmaModel(config, gemmaCfg, weights);
                }

                case ModelType.Phi:
                {
                    PhiModel.DebugEnabled = debugEnabled;
                    var phiCfg = new PhiConfig(config);
                    var weights = new PhiWeights(weightLoader, phiCfg);
                    return new PhiModel(config, weights);
                }

                default:
                    throw new NotSupportedException("Unsupported model architecture: " + modelType);
            }
        }

        /// <summary>
        /// Create the appropriate chat template for the detected architecture.
        /// </summary>
        public static IChatTemplate CreateChatTemplate(ModelType modelType)
        {
            switch (modelType)
            {
                case ModelType.Llama:
                    return new LlamaChatTemplate();
                case ModelType.Qwen:
                    return new QwenChatTemplate();
                case ModelType.Mistral:
                    return new MistralChatTemplate();
                case ModelType.Gemma:
                    return new GemmaChatTemplate();
                case ModelType.Phi:
                    return new PhiChatTemplate();
                default:
                    return new LlamaChatTemplate();
            }
        }

        /// <summary>
        /// Get the appropriate stop strings for the detected architecture.
        /// </summary>
        public static string[] GetStopStrings(ModelType modelType)
        {
            switch (modelType)
            {
                case ModelType.Llama:
                    return new[] { "</s>" };
                case ModelType.Qwen:
                    return new[] { "<|im_end|>", "<|endoftext|>" };
                case ModelType.Mistral:
                    return new[] { "</s>" };
                case ModelType.Gemma:
                    return new[] { "<end_of_turn>", "</s>" };
                case ModelType.Phi:
                    return new[] { "<|end|>", "<|endoftext|>" };
                default:
                    return new[] { "</s>" };
            }
        }
    }
}
