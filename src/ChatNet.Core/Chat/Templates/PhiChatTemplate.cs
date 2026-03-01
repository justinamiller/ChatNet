using System.Text;
using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Chat.Templates
{
    /// <summary>
    /// Phi-3 chat template format (ChatML-based):
    /// &lt;|system|&gt;\n{content}&lt;|end|&gt;\n
    /// &lt;|user|&gt;\n{content}&lt;|end|&gt;\n
    /// &lt;|assistant|&gt;\n
    /// </summary>
    public sealed class PhiChatTemplate : IChatTemplate
    {
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
                        sb.Append("<|system|>\n");
                        sb.Append(msg.Content);
                        sb.Append("<|end|>\n");
                        break;
                    case ChatRole.User:
                        sb.Append("<|user|>\n");
                        sb.Append(msg.Content);
                        sb.Append("<|end|>\n");
                        break;
                    case ChatRole.Assistant:
                        sb.Append("<|assistant|>\n");
                        sb.Append(msg.Content);
                        sb.Append("<|end|>\n");
                        break;
                }
            }

            if (!hasSystem)
            {
                string existing = sb.ToString();
                sb.Clear();
                sb.Append("<|system|>\nYou are a helpful assistant.<|end|>\n");
                sb.Append(existing);
            }

            sb.Append("<|assistant|>\n");
            return sb.ToString();
        }
    }
}
