using System;
using System.Text;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat;

namespace ChatNet.Core.Chat.Templates
{
    /// <summary>
    /// TinyLlama / Zephyr chat template format:
    /// &lt;|system|&gt;\n{content}&lt;/s&gt;\n
    /// &lt;|user|&gt;\n{content}&lt;/s&gt;\n
    /// &lt;|assistant|&gt;\n
    /// </summary>
    public sealed class LlamaChatTemplate : IChatTemplate
    {
        private static readonly string SystemTag = "<|system|>";
        private static readonly string UserTag = "<|user|>";
        private static readonly string AssistantTag = "<|assistant|>";
        private static readonly string EndTurn = "</s>";

        public string FormatPrompt(ChatMessage[] messages)
        {
            var sb = new StringBuilder(512);

            bool hasSystem = false;

            for (int i = 0; i < messages.Length; i++)
            {
                ChatMessage msg = messages[i];
                switch (msg.Role)
                {
                    case ChatRole.System:
                        hasSystem = true;
                        sb.Append(SystemTag);
                        sb.Append('\n');
                        sb.Append(msg.Content);
                        sb.Append(EndTurn);
                        sb.Append('\n');
                        break;
                    case ChatRole.User:
                        sb.Append(UserTag);
                        sb.Append('\n');
                        sb.Append(msg.Content);
                        sb.Append(EndTurn);
                        sb.Append('\n');
                        break;
                    case ChatRole.Assistant:
                        sb.Append(AssistantTag);
                        sb.Append('\n');
                        sb.Append(msg.Content);
                        sb.Append(EndTurn);
                        sb.Append('\n');
                        break;
                }
            }

            // If no system message was provided, add a default one
            if (!hasSystem)
            {
                string existing = sb.ToString();
                sb.Clear();
                sb.Append(SystemTag);
                sb.Append('\n');
                sb.Append("You are a helpful assistant.");
                sb.Append(EndTurn);
                sb.Append('\n');
                sb.Append(existing);
            }

            // Always end with the assistant tag to prompt generation
            sb.Append(AssistantTag);
            sb.Append('\n');

            return sb.ToString();
        }
    }
}
