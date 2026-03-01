using System.Text;
using ChatNet.Core.Abstractions;

namespace ChatNet.Core.Chat.Templates
{
    /// <summary>
    /// Gemma chat template format:
    /// &lt;start_of_turn&gt;user\n{content}&lt;end_of_turn&gt;\n
    /// &lt;start_of_turn&gt;model\n
    /// </summary>
    public sealed class GemmaChatTemplate : IChatTemplate
    {
        public string FormatPrompt(ChatMessage[] messages)
        {
            var sb = new StringBuilder(512);

            for (int i = 0; i < messages.Length; i++)
            {
                ChatMessage msg = messages[i];
                switch (msg.Role)
                {
                    case ChatRole.System:
                        // Gemma doesn't have a formal system role; prepend as user context
                        sb.Append("<start_of_turn>user\n");
                        sb.Append(msg.Content);
                        sb.Append("<end_of_turn>\n");
                        break;
                    case ChatRole.User:
                        sb.Append("<start_of_turn>user\n");
                        sb.Append(msg.Content);
                        sb.Append("<end_of_turn>\n");
                        break;
                    case ChatRole.Assistant:
                        sb.Append("<start_of_turn>model\n");
                        sb.Append(msg.Content);
                        sb.Append("<end_of_turn>\n");
                        break;
                }
            }

            sb.Append("<start_of_turn>model\n");
            return sb.ToString();
        }
    }
}
