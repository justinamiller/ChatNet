using System;
using System.Diagnostics;
using System.Text;
using ChatNet.Core;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat;
using ChatNet.Core.Chat.Templates;
using ChatNet.Core.Samplers;

namespace ChatNet.Cli
{
    /// <summary>
    /// Token output handler that avoids closures by holding state in a class instance.
    /// </summary>
    internal sealed class TokenOutputHandler
    {
        public readonly StringBuilder ResponseBuilder = new StringBuilder(512);
        public bool Cancelled;

        public void OnToken(string token)
        {
            if (!Cancelled)
            {
                Console.Write(token);
                Console.Out.Flush();
                ResponseBuilder.Append(token);
            }
        }

        public void Reset()
        {
            ResponseBuilder.Clear();
            Cancelled = false;
        }
    }

    internal static class Program
    {
        private static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;

            // Parse command-line arguments
            string? modelPath = null;
            string? prompt = null;
            int maxTokens = 128;
            float temperature = 0.0f;
            int topK = 40;
            float topP = 0.9f;
            bool interactive = false;
            bool debug = false;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--debug":
                        debug = true;
                        break;
                    case "--model":
                    case "-m":
                        if (i + 1 < args.Length) modelPath = args[++i];
                        break;
                    case "--prompt":
                    case "-p":
                        if (i + 1 < args.Length) prompt = args[++i];
                        break;
                    case "--max-tokens":
                    case "-n":
                        if (i + 1 < args.Length) maxTokens = int.Parse(args[++i]);
                        break;
                    case "--temp":
                    case "-t":
                        if (i + 1 < args.Length) temperature = float.Parse(args[++i]);
                        break;
                    case "--top-k":
                        if (i + 1 < args.Length) topK = int.Parse(args[++i]);
                        break;
                    case "--top-p":
                        if (i + 1 < args.Length) topP = float.Parse(args[++i]);
                        break;
                    case "--interactive":
                    case "-i":
                        interactive = true;
                        break;
                    case "--help":
                    case "-h":
                        PrintHelp();
                        return 0;
                }
            }

            if (modelPath == null)
            {
                Console.Error.WriteLine("Error: --model path is required.");
                Console.Error.WriteLine("Use --help for usage information.");
                return 1;
            }

            // Banner
            Console.WriteLine("ChatNet v0.1.0 — Pure C# LLM Inference");
            Console.WriteLine("========================================");
            Console.WriteLine();

            // Load model
            Console.Write("Loading model: " + modelPath + " ... ");
            Console.Out.Flush();

            InferenceEngine.DebugEnabled = debug;

            InferenceEngine engine;
            try
            {
                engine = InferenceEngine.LoadFromGguf(modelPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine("Error loading model: " + ex.Message);
                return 1;
            }

            ModelConfig config = engine.Config;
            Console.WriteLine("Done!");
            Console.WriteLine("Model: " + config.ModelName + " | Arch: " + config.Architecture +
                " | Layers: " + config.LayerCount + " | Dim: " + config.EmbeddingDim);
            Console.WriteLine("Vocab: " + config.VocabSize + " | Heads: " + config.AttentionHeadCount +
                "/" + config.KeyValueHeadCount + " | FFN: " + config.FeedForwardDim);
            Console.WriteLine("Loaded in " + engine.LoadTimeMs + "ms");
            Console.WriteLine();

            using (engine)
            {
                // Create sampler
                ISampler sampler;
                if (temperature <= 0f)
                {
                    sampler = new GreedySampler();
                }
                else
                {
                    sampler = new TemperatureSampler(temperature, topK, topP);
                }

                if (prompt != null && !interactive)
                {
                    // Single prompt mode
                    RunSinglePrompt(engine, sampler, prompt, maxTokens);
                }
                else
                {
                    // Interactive chat mode
                    if (prompt != null)
                    {
                        RunSinglePrompt(engine, sampler, prompt, maxTokens);
                        Console.WriteLine();
                    }
                    RunInteractive(engine, sampler, maxTokens);
                }
            }

            return 0;
        }

        // Shared handler instance to avoid per-call allocations
        private static readonly TokenOutputHandler s_handler = new TokenOutputHandler();

        private static void RunSinglePrompt(InferenceEngine engine, ISampler sampler, string prompt, int maxTokens)
        {
            var session = new ChatSession(new LlamaChatTemplate());
            session.AddUserMessage(prompt);

            Console.Write("> ");
            Console.WriteLine(prompt);
            Console.WriteLine();

            s_handler.Reset();
            var sw = Stopwatch.StartNew();

            int tokenCount = engine.GenerateChat(session, sampler, maxTokens, s_handler.OnToken);

            sw.Stop();
            Console.WriteLine();
            Console.WriteLine();

            double tokensPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
            Console.WriteLine("[" + tokenCount + " tokens | " + tokensPerSec.ToString("F1") +
                " tok/s | " + sw.Elapsed.TotalSeconds.ToString("F1") + "s]");
        }

        // Static cancel handler that references the shared handler
        private static void OnCancelKeyPress(object? sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
            s_handler.Cancelled = true;
        }

        private static void RunInteractive(InferenceEngine engine, ISampler sampler, int maxTokens)
        {
            Console.WriteLine("Interactive chat mode. Type 'exit' or 'quit' to stop. Ctrl+C to cancel generation.");
            Console.WriteLine();

            var session = new ChatSession(new LlamaChatTemplate());

            while (true)
            {
                Console.Write("> ");
                Console.Out.Flush();
                string? input = Console.ReadLine();

                if (input == null) break;

                string trimmed = input.Trim();
                if (trimmed.Length == 0) continue;
                if (trimmed == "exit" || trimmed == "quit") break;
                if (trimmed == "/clear")
                {
                    session.Clear();
                    Console.WriteLine("Conversation cleared.");
                    continue;
                }

                session.AddUserMessage(trimmed);

                Console.WriteLine();
                s_handler.Reset();
                var sw = Stopwatch.StartNew();

                Console.CancelKeyPress += OnCancelKeyPress;

                int tokenCount;
                try
                {
                    tokenCount = engine.GenerateChat(session, sampler, maxTokens, s_handler.OnToken);
                }
                finally
                {
                    Console.CancelKeyPress -= OnCancelKeyPress;
                }

                sw.Stop();
                Console.WriteLine();

                // Add assistant response to history
                session.AddAssistantMessage(s_handler.ResponseBuilder.ToString());

                double tokensPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
                Console.WriteLine("[" + tokenCount + " tokens | " + tokensPerSec.ToString("F1") +
                    " tok/s | " + sw.Elapsed.TotalSeconds.ToString("F1") + "s]");
                Console.WriteLine();
            }

            Console.WriteLine("Bye!");
        }

        private static void PrintHelp()
        {
            Console.WriteLine("ChatNet CLI — Pure C# LLM Inference Engine");
            Console.WriteLine();
            Console.WriteLine("Usage:");
            Console.WriteLine("  dotnet run --project src/ChatNet.Cli -- [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --model, -m <path>       Path to GGUF model file (required)");
            Console.WriteLine("  --prompt, -p <text>      Input prompt text");
            Console.WriteLine("  --max-tokens, -n <int>   Maximum tokens to generate (default: 128)");
            Console.WriteLine("  --temp, -t <float>       Temperature for sampling (0.0 = greedy, default: 0.0)");
            Console.WriteLine("  --top-k <int>            Top-K sampling (default: 40)");
            Console.WriteLine("  --top-p <float>          Top-P / nucleus sampling (default: 0.9)");
            Console.WriteLine("  --interactive, -i        Enter interactive chat mode");
            Console.WriteLine("  --help, -h               Show this help message");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  dotnet run -- --model model.gguf --prompt \"What is the capital of France?\"");
            Console.WriteLine("  dotnet run -- --model model.gguf --interactive");
            Console.WriteLine("  dotnet run -- --model model.gguf --prompt \"Hello\" --temp 0.7 --max-tokens 256");
        }
    }
}
