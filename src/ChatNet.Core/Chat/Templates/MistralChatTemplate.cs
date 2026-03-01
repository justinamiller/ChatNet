using System.Text;
using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Chat.Templates
{
    /// <summary>
    /// Mistral Instruct chat template format:
    /// [INST] {system}\n{user_message} [/INST]
    /// </summary>
    public sealed class MistralChatTemplate : IChatTemplate
    {
        public string FormatPrompt(ChatMessage[] messages)
        {
            var sb = new StringBuilder(512);

            string? systemContent = null;
            for (int i = 0; i < messages.Length; i++)
            {
                ChatMessage msg = messages[i];
                switch (msg.Role)
                {
                    case ChatRole.System:
                        systemContent = msg.Content;
                        break;
                    case ChatRole.User:
                        sb.Append("[INST] ");
                        if (systemContent != null)
                        {
                            sb.Append(systemContent);
                            sb.Append("\n\n");
                            systemContent = null;
                        }
                        sb.Append(msg.Content);
                        sb.Append(" [/INST]");
                        break;
                    case ChatRole.Assistant:
                        sb.Append(msg.Content);
                        sb.Append("</s>");
                        break;
                }
            }

            return sb.ToString();
        }
    }
}
