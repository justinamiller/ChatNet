using System;
using System.Collections.Generic;
using ChatNet.Core.Abstractions;
using ChatNet.Core.Chat.Templates;

namespace ChatNet.Core.Chat
{
    /// <summary>
    /// Manages conversation context and history.
    /// Builds prompts using the appropriate chat template.
    /// </summary>
    public sealed class ChatSession
    {
        private readonly List<ChatMessage> _messages;
        private readonly IChatTemplate _template;

        public ChatSession(IChatTemplate? template = null)
        {
            _template = template ?? new LlamaChatTemplate();
            _messages = new List<ChatMessage>(32);
        }

        /// <summary>Add a system message.</summary>
        public void AddSystemMessage(string content)
        {
            _messages.Add(new ChatMessage(ChatRole.System, content));
        }

        /// <summary>Add a user message.</summary>
        public void AddUserMessage(string content)
        {
            _messages.Add(new ChatMessage(ChatRole.User, content));
        }

        /// <summary>Add an assistant message (for history tracking).</summary>
        public void AddAssistantMessage(string content)
        {
            _messages.Add(new ChatMessage(ChatRole.Assistant, content));
        }

        /// <summary>Build the formatted prompt text for the current conversation.</summary>
        public string BuildPrompt()
        {
            return _template.FormatPrompt(_messages.ToArray());
        }

        /// <summary>Get all messages.</summary>
        public ChatMessage[] GetMessages()
        {
            return _messages.ToArray();
        }

        /// <summary>Clear conversation history.</summary>
        public void Clear()
        {
            _messages.Clear();
        }

        /// <summary>Number of messages in the conversation.</summary>
        public int MessageCount => _messages.Count;
    }
}
