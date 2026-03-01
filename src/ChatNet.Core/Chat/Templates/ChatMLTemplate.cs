using System;
using System.Text;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat;

namespace ChatNet.Core.Chat.Templates
{
    /// <summary>
    /// ChatML template format (for future models like Phi, etc.):
    /// &lt;|im_start|&gt;system\n{content}&lt;|im_end|&gt;\n
    /// &lt;|im_start|&gt;user\n{content}&lt;|im_end|&gt;\n
    /// &lt;|im_start|&gt;assistant\n
    /// </summary>
    public sealed class ChatMLTemplate : IChatTemplate
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
                        sb.Append("<|im_start|>system\n");
                        sb.Append(msg.Content);
                        sb.Append("<|im_end|>\n");
                        break;
                    case ChatRole.User:
                        sb.Append("<|im_start|>user\n");
                        sb.Append(msg.Content);
                        sb.Append("<|im_end|>\n");
                        break;
                    case ChatRole.Assistant:
                        sb.Append("<|im_start|>assistant\n");
                        sb.Append(msg.Content);
                        sb.Append("<|im_end|>\n");
                        break;
                }
            }

            if (!hasSystem)
            {
                string existing = sb.ToString();
                sb.Clear();
                sb.Append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
                sb.Append(existing);
            }

            sb.Append("<|im_start|>assistant\n");
            return sb.ToString();
        }
    }
}
