package xai

// ModerationError is returned by the xAI video model when the API rejects
// generated content due to content moderation policy.
type ModerationError struct {
	Code    string
	Message string
}

// Error implements the error interface.
func (e *ModerationError) Error() string {
	if e.Code != "" {
		return "xai: moderation rejection [" + e.Code + "]: " + e.Message
	}
	return "xai: moderation rejection: " + e.Message
}
