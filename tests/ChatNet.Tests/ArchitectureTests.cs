using System;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat;
using ChatNet.Core.Chat.Templates;
using ChatNet.Core.Models;
using ChatNet.Core.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ChatNet.Tests
{
    /// <summary>
    /// Tests for multi-architecture inference support.
    /// Validates factory routing, chat templates, tensor math additions,
    /// and architecture-specific forward pass behavior.
    /// </summary>
    [TestClass]
    public class ArchitectureTests
    {
        #region Architecture Detection Tests

        [TestMethod]
        public void DetectArchitecture_Llama_ReturnsLlama()
        {
            Assert.AreEqual(ModelType.Llama, ModelFactory.DetectArchitecture("llama"));
        }

        [TestMethod]
        public void DetectArchitecture_Qwen2_ReturnsQwen()
        {
            Assert.AreEqual(ModelType.Qwen, ModelFactory.DetectArchitecture("qwen2"));
        }

        [TestMethod]
        public void DetectArchitecture_Qwen_ReturnsQwen()
        {
            Assert.AreEqual(ModelType.Qwen, ModelFactory.DetectArchitecture("qwen"));
        }

        [TestMethod]
        public void DetectArchitecture_Mistral_ReturnsMistral()
        {
            Assert.AreEqual(ModelType.Mistral, ModelFactory.DetectArchitecture("mistral"));
        }

        [TestMethod]
        public void DetectArchitecture_Gemma_ReturnsGemma()
        {
            Assert.AreEqual(ModelType.Gemma, ModelFactory.DetectArchitecture("gemma"));
        }

        [TestMethod]
        public void DetectArchitecture_Gemma2_ReturnsGemma()
        {
            Assert.AreEqual(ModelType.Gemma, ModelFactory.DetectArchitecture("gemma2"));
        }

        [TestMethod]
        public void DetectArchitecture_Phi2_ReturnsPhi()
        {
            Assert.AreEqual(ModelType.Phi, ModelFactory.DetectArchitecture("phi2"));
        }

        [TestMethod]
        public void DetectArchitecture_Phi3_ReturnsPhi()
        {
            Assert.AreEqual(ModelType.Phi, ModelFactory.DetectArchitecture("phi3"));
        }

        [TestMethod]
        public void DetectArchitecture_Phi_ReturnsPhi()
        {
            Assert.AreEqual(ModelType.Phi, ModelFactory.DetectArchitecture("phi"));
        }

        [TestMethod]
        public void DetectArchitecture_CaseInsensitive()
        {
            Assert.AreEqual(ModelType.Llama, ModelFactory.DetectArchitecture("LLAMA"));
            Assert.AreEqual(ModelType.Qwen, ModelFactory.DetectArchitecture("Qwen2"));
            Assert.AreEqual(ModelType.Mistral, ModelFactory.DetectArchitecture("MISTRAL"));
            Assert.AreEqual(ModelType.Gemma, ModelFactory.DetectArchitecture("Gemma"));
            Assert.AreEqual(ModelType.Phi, ModelFactory.DetectArchitecture("PHI3"));
        }

        [TestMethod]
        public void DetectArchitecture_Unknown_DefaultsToLlama()
        {
            Assert.AreEqual(ModelType.Llama, ModelFactory.DetectArchitecture("unknown_arch"));
        }

        #endregion

        #region Chat Template Tests

        [TestMethod]
        public void CreateChatTemplate_Llama_ReturnsCorrectType()
        {
            IChatTemplate template = ModelFactory.CreateChatTemplate(ModelType.Llama);
            Assert.IsInstanceOfType(template, typeof(LlamaChatTemplate));
        }

        [TestMethod]
        public void CreateChatTemplate_Qwen_ReturnsQwenTemplate()
        {
            IChatTemplate template = ModelFactory.CreateChatTemplate(ModelType.Qwen);
            Assert.IsInstanceOfType(template, typeof(QwenChatTemplate));
        }

        [TestMethod]
        public void CreateChatTemplate_Mistral_ReturnsMistralTemplate()
        {
            IChatTemplate template = ModelFactory.CreateChatTemplate(ModelType.Mistral);
            Assert.IsInstanceOfType(template, typeof(MistralChatTemplate));
        }

        [TestMethod]
        public void CreateChatTemplate_Gemma_ReturnsGemmaTemplate()
        {
            IChatTemplate template = ModelFactory.CreateChatTemplate(ModelType.Gemma);
            Assert.IsInstanceOfType(template, typeof(GemmaChatTemplate));
        }

        [TestMethod]
        public void CreateChatTemplate_Phi_ReturnsPhiTemplate()
        {
            IChatTemplate template = ModelFactory.CreateChatTemplate(ModelType.Phi);
            Assert.IsInstanceOfType(template, typeof(PhiChatTemplate));
        }

        [TestMethod]
        public void QwenTemplate_FormatsCorrectly()
        {
            var template = new QwenChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.User, "Hello"),
            };
            string prompt = template.FormatPrompt(messages);

            Assert.IsTrue(prompt.Contains("<|im_start|>system"), "Should have system message");
            Assert.IsTrue(prompt.Contains("<|im_start|>user"), "Should have user tag");
            Assert.IsTrue(prompt.Contains("Hello"), "Should contain user message");
            Assert.IsTrue(prompt.Contains("<|im_end|>"), "Should have end tags");
            Assert.IsTrue(prompt.EndsWith("<|im_start|>assistant\n"), "Should end with assistant tag");
        }

        [TestMethod]
        public void MistralTemplate_FormatsCorrectly()
        {
            var template = new MistralChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.User, "Hello"),
            };
            string prompt = template.FormatPrompt(messages);

            Assert.IsTrue(prompt.Contains("[INST]"), "Should have INST tag");
            Assert.IsTrue(prompt.Contains("Hello"), "Should contain user message");
            Assert.IsTrue(prompt.Contains("[/INST]"), "Should have closing INST tag");
        }

        [TestMethod]
        public void MistralTemplate_WithSystem_FormatsCorrectly()
        {
            var template = new MistralChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.System, "You are helpful."),
                new ChatMessage(ChatRole.User, "Hello"),
            };
            string prompt = template.FormatPrompt(messages);

            Assert.IsTrue(prompt.Contains("[INST] You are helpful."), "Should include system in INST");
            Assert.IsTrue(prompt.Contains("Hello"), "Should contain user message");
        }

        [TestMethod]
        public void GemmaTemplate_FormatsCorrectly()
        {
            var template = new GemmaChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.User, "Hello"),
            };
            string prompt = template.FormatPrompt(messages);

            Assert.IsTrue(prompt.Contains("<start_of_turn>user"), "Should have user turn start");
            Assert.IsTrue(prompt.Contains("Hello"), "Should contain user message");
            Assert.IsTrue(prompt.Contains("<end_of_turn>"), "Should have end turn tag");
            Assert.IsTrue(prompt.EndsWith("<start_of_turn>model\n"), "Should end with model turn");
        }

        [TestMethod]
        public void PhiTemplate_FormatsCorrectly()
        {
            var template = new PhiChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.User, "Hello"),
            };
            string prompt = template.FormatPrompt(messages);

            Assert.IsTrue(prompt.Contains("<|system|>"), "Should have system tag");
            Assert.IsTrue(prompt.Contains("<|user|>"), "Should have user tag");
            Assert.IsTrue(prompt.Contains("Hello"), "Should contain user message");
            Assert.IsTrue(prompt.Contains("<|end|>"), "Should have end tags");
            Assert.IsTrue(prompt.EndsWith("<|assistant|>\n"), "Should end with assistant tag");
        }

        #endregion

        #region Stop String Tests

        [TestMethod]
        public void GetStopStrings_ReturnsArchitectureSpecific()
        {
            string[] llamaStops = ModelFactory.GetStopStrings(ModelType.Llama);
            Assert.IsTrue(Array.IndexOf(llamaStops, "</s>") >= 0);

            string[] qwenStops = ModelFactory.GetStopStrings(ModelType.Qwen);
            Assert.IsTrue(Array.IndexOf(qwenStops, "<|im_end|>") >= 0);

            string[] mistralStops = ModelFactory.GetStopStrings(ModelType.Mistral);
            Assert.IsTrue(Array.IndexOf(mistralStops, "</s>") >= 0);

            string[] gemmaStops = ModelFactory.GetStopStrings(ModelType.Gemma);
            Assert.IsTrue(Array.IndexOf(gemmaStops, "<end_of_turn>") >= 0);

            string[] phiStops = ModelFactory.GetStopStrings(ModelType.Phi);
            Assert.IsTrue(Array.IndexOf(phiStops, "<|end|>") >= 0);
        }

        #endregion

        #region GELU Tests

        [TestMethod]
        public void GeluElementwiseMul_ZeroInput_ReturnsZero()
        {
            float[] gate = new float[32];
            float[] up = new float[32];
            for (int i = 0; i < 32; i++) up[i] = 1.0f;

            TensorMath.GeluElementwiseMul(gate, up, 32);

            for (int i = 0; i < 32; i++)
            {
                Assert.AreEqual(0.0f, gate[i], 0.001f,
                    "GELU(0) * 1.0 should be 0");
            }
        }

        [TestMethod]
        public void GeluElementwiseMul_PositiveInput_ReturnsPositive()
        {
            float[] gate = new float[32];
            float[] up = new float[32];
            for (int i = 0; i < 32; i++)
            {
                gate[i] = 2.0f;
                up[i] = 1.0f;
            }

            TensorMath.GeluElementwiseMul(gate, up, 32);

            for (int i = 0; i < 32; i++)
            {
                Assert.IsTrue(gate[i] > 1.9f && gate[i] < 2.1f,
                    "GELU(2.0) * 1.0 should be close to 2.0");
            }
        }

        [TestMethod]
        public void GeluElementwiseMul_NegativeInput_ReturnsNearZero()
        {
            float[] gate = new float[32];
            float[] up = new float[32];
            for (int i = 0; i < 32; i++)
            {
                gate[i] = -3.0f;
                up[i] = 1.0f;
            }

            TensorMath.GeluElementwiseMul(gate, up, 32);

            for (int i = 0; i < 32; i++)
            {
                Assert.IsTrue(gate[i] > -0.1f && gate[i] < 0.1f,
                    "GELU(-3.0) should be close to 0");
            }
        }

        [TestMethod]
        public void GeluElementwiseMul_MultipliesWithUp()
        {
            float[] gate = new float[32];
            float[] up = new float[32];
            for (int i = 0; i < 32; i++)
            {
                gate[i] = 1.0f;
                up[i] = 2.0f;
            }

            TensorMath.GeluElementwiseMul(gate, up, 32);

            // GELU(1.0) ~= 0.8412, so result ~= 1.6824
            for (int i = 0; i < 32; i++)
            {
                Assert.IsTrue(gate[i] > 1.5f && gate[i] < 1.9f,
                    $"GELU(1.0) * 2.0 ~= 1.68, got {gate[i]}");
            }
        }

        [TestMethod]
        public unsafe void GeluElementwiseMul_HandlesNonAlignedLengths()
        {
            // Test with lengths that don't align to the 4x unroll
            int[] testLengths = new[] { 1, 2, 3, 5, 7, 15, 17, 33 };
            for (int li = 0; li < testLengths.Length; li++)
            {
                int len = testLengths[li];
                float[] gate = new float[len];
                float[] up = new float[len];
                for (int i = 0; i < len; i++)
                {
                    gate[i] = 1.0f;
                    up[i] = 1.0f;
                }

                TensorMath.GeluElementwiseMul(gate, up, len);

                for (int i = 0; i < len; i++)
                {
                    Assert.IsFalse(float.IsNaN(gate[i]), $"NaN at index {i} for length {len}");
                    Assert.IsFalse(float.IsInfinity(gate[i]), $"Inf at index {i} for length {len}");
                }
            }
        }

        #endregion

        #region RmsNormWithOffset Tests

        [TestMethod]
        public void RmsNormWithOffset_OnesWeight_AppliesOffset()
        {
            int dim = 64;
            float[] input = new float[dim];
            float[] weight = new float[dim]; // zero weights -> +1 offset = 1.0
            float[] output = new float[dim];

            for (int i = 0; i < dim; i++)
            {
                input[i] = 1.0f;
            }

            TensorMath.RmsNormWithOffset(input, weight, output, 1e-5f);

            // With weight+1 = 1.0 and uniform input of 1.0:
            // RMS = sqrt(1.0 + eps) ~= 1.0, scale = 1.0/1.0 = 1.0
            // output[i] = 1.0 * 1.0 * 1.0 = 1.0
            for (int i = 0; i < dim; i++)
            {
                Assert.AreEqual(1.0f, output[i], 0.01f,
                    $"RmsNormWithOffset element {i} mismatch");
            }
        }

        [TestMethod]
        public void RmsNormWithOffset_ProducesFiniteValues()
        {
            int dim = 256;
            float[] input = new float[dim];
            float[] weight = new float[dim];
            float[] output = new float[dim];

            for (int i = 0; i < dim; i++)
            {
                input[i] = (float)(i - dim / 2) / dim;
                weight[i] = 0.01f * i;
            }

            TensorMath.RmsNormWithOffset(input, weight, output, 1e-5f);

            for (int i = 0; i < dim; i++)
            {
                Assert.IsFalse(float.IsNaN(output[i]), $"NaN at index {i}");
                Assert.IsFalse(float.IsInfinity(output[i]), $"Inf at index {i}");
            }
        }

        [TestMethod]
        public void RmsNormWithOffset_DiffersFromRmsNorm()
        {
            int dim = 64;
            float[] input = new float[dim];
            float[] weight = new float[dim];
            float[] outputStandard = new float[dim];
            float[] outputOffset = new float[dim];

            for (int i = 0; i < dim; i++)
            {
                input[i] = 1.0f;
                weight[i] = 0.5f; // Non-zero weight to see difference
            }

            TensorMath.RmsNorm(input, weight, outputStandard, 1e-5f);
            TensorMath.RmsNormWithOffset(input, weight, outputOffset, 1e-5f);

            // Standard uses weight[i]=0.5, offset uses weight[i]+1=1.5
            // Outputs should differ
            bool anyDifferent = false;
            for (int i = 0; i < dim; i++)
            {
                if (MathF.Abs(outputStandard[i] - outputOffset[i]) > 0.01f)
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.IsTrue(anyDifferent, "RmsNormWithOffset should differ from RmsNorm when weight != 0");
        }

        #endregion

        #region ModelType Enum Tests

        [TestMethod]
        public void ModelType_HasAllExpectedValues()
        {
            Assert.IsTrue(Enum.IsDefined(typeof(ModelType), ModelType.Llama));
            Assert.IsTrue(Enum.IsDefined(typeof(ModelType), ModelType.Qwen));
            Assert.IsTrue(Enum.IsDefined(typeof(ModelType), ModelType.Mistral));
            Assert.IsTrue(Enum.IsDefined(typeof(ModelType), ModelType.Gemma));
            Assert.IsTrue(Enum.IsDefined(typeof(ModelType), ModelType.Phi));
        }

        #endregion

        #region ModelConfig Tests

        [TestMethod]
        public void ModelConfig_SoftcapDefaults_AreZero()
        {
            var config = new ModelConfig();
            Assert.AreEqual(0f, config.AttnLogitSoftcap);
            Assert.AreEqual(0f, config.FinalLogitSoftcap);
        }

        [TestMethod]
        public void ModelConfig_RopeFreqBaseDefault_Is10000()
        {
            var config = new ModelConfig();
            Assert.AreEqual(10000.0f, config.RopeFreqBase);
        }

        #endregion

        #region GgufMetadata Float Conversion Tests

        [TestMethod]
        public void GgufMetadata_GetFloat32_HandlesIntTypes()
        {
            var meta = new ChatNet.Core.Gguf.GgufMetadata();
            meta.Set("test_uint", (uint)1000000);
            meta.Set("test_int", (int)500000);

            Assert.AreEqual(1000000f, meta.GetFloat32("test_uint"), 1f);
            Assert.AreEqual(500000f, meta.GetFloat32("test_int"), 1f);
        }

        [TestMethod]
        public void GgufMetadata_GetFloat32_HandlesFloatAndDouble()
        {
            var meta = new ChatNet.Core.Gguf.GgufMetadata();
            meta.Set("test_float", 1000000.0f);
            meta.Set("test_double", 1000000.0);

            Assert.AreEqual(1000000f, meta.GetFloat32("test_float"), 1f);
            Assert.AreEqual(1000000f, meta.GetFloat32("test_double"), 1f);
        }

        #endregion

        #region Chat Template Multi-Turn Tests

        [TestMethod]
        public void QwenTemplate_MultiTurn_CorrectFormat()
        {
            var template = new QwenChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.System, "Be helpful."),
                new ChatMessage(ChatRole.User, "Hi"),
                new ChatMessage(ChatRole.Assistant, "Hello!"),
                new ChatMessage(ChatRole.User, "How are you?"),
            };
            string prompt = template.FormatPrompt(messages);

            // Verify ordering and structure
            int sysPos = prompt.IndexOf("<|im_start|>system");
            int user1Pos = prompt.IndexOf("Hi");
            int asst1Pos = prompt.IndexOf("Hello!");
            int user2Pos = prompt.IndexOf("How are you?");

            Assert.IsTrue(sysPos < user1Pos, "System before first user");
            Assert.IsTrue(user1Pos < asst1Pos, "First user before assistant");
            Assert.IsTrue(asst1Pos < user2Pos, "Assistant before second user");
        }

        [TestMethod]
        public void GemmaTemplate_MultiTurn_CorrectFormat()
        {
            var template = new GemmaChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.User, "Hi"),
                new ChatMessage(ChatRole.Assistant, "Hello!"),
                new ChatMessage(ChatRole.User, "Question"),
            };
            string prompt = template.FormatPrompt(messages);

            int u1 = prompt.IndexOf("Hi");
            int a1 = prompt.IndexOf("Hello!");
            int u2 = prompt.IndexOf("Question");

            Assert.IsTrue(u1 < a1, "First user before assistant");
            Assert.IsTrue(a1 < u2, "Assistant before second user");
            Assert.IsTrue(prompt.EndsWith("<start_of_turn>model\n"), "Ends with model turn");
        }

        [TestMethod]
        public void PhiTemplate_MultiTurn_CorrectFormat()
        {
            var template = new PhiChatTemplate();
            var messages = new[]
            {
                new ChatMessage(ChatRole.User, "Hi"),
                new ChatMessage(ChatRole.Assistant, "Hello!"),
                new ChatMessage(ChatRole.User, "Question"),
            };
            string prompt = template.FormatPrompt(messages);

            Assert.IsTrue(prompt.Contains("<|system|>"), "Has default system");
            Assert.IsTrue(prompt.Contains("<|user|>\nHi<|end|>"), "Has first user");
            Assert.IsTrue(prompt.Contains("<|assistant|>\nHello!<|end|>"), "Has assistant");
            Assert.IsTrue(prompt.EndsWith("<|assistant|>\n"), "Ends with assistant tag");
        }

        #endregion
    }
}
