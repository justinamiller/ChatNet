namespace ChatNet.Core.Chat
{
    /// <summary>
    /// A single chat message with role and content.
    /// </summary>
    public readonly struct ChatMessage
    {
        public ChatRole Role { get; }
        public string Content { get; }

        public ChatMessage(ChatRole role, string content)
        {
            Role = role;
            Content = content;
        }
    }

    /// <summary>
    /// Chat message roles.
    /// </summary>
    public enum ChatRole
    {
        System,
        User,
        Assistant,
    }
}
