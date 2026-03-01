using System;
using ChatNet.Core.Chat;

namespace ChatNet.Core.Abstractions
{
    /// <summary>
    /// Chat template abstraction for formatting conversations into model prompt format.
    /// </summary>
    public interface IChatTemplate
    {
        /// <summary>Format a conversation into the model's expected prompt format.</summary>
        string FormatPrompt(ChatMessage[] messages);
    }
}
