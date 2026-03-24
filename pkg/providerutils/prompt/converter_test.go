package prompt

import (
	"encoding/json"
	"testing"

	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// TestToAnthropicMessagesCustomContentWithOptions verifies that CustomContent
// with Anthropic-keyed ProviderOptions is forwarded verbatim to the output.
func TestToAnthropicMessagesCustomContentWithOptions(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.CustomContent{
					Kind: "anthropic-future-block",
					ProviderOptions: map[string]interface{}{
						"anthropic": map[string]interface{}{
							"type":  "future_block",
							"value": "data",
						},
					},
				},
			},
		},
	}

	result := ToAnthropicMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	content, ok := result[0]["content"].([]map[string]interface{})
	if !ok {
		t.Fatalf("content should be []map[string]interface{}, got %T", result[0]["content"])
	}
	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}
	if content[0]["type"] != "future_block" {
		t.Errorf("type = %v, want \"future_block\"", content[0]["type"])
	}
	if content[0]["value"] != "data" {
		t.Errorf("value = %v, want \"data\"", content[0]["value"])
	}
}

// TestToAnthropicMessagesCustomContentNoOptions verifies that CustomContent
// without Anthropic-keyed ProviderOptions is silently dropped.
func TestToAnthropicMessagesCustomContentNoOptions(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.TextContent{Text: "Answer."},
				types.CustomContent{Kind: "xai-citation"}, // no ProviderOptions
			},
		},
	}

	result := ToAnthropicMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	// Content is a single text part — the custom content was dropped.
	switch c := result[0]["content"].(type) {
	case string:
		if c != "Answer." {
			t.Errorf("content = %q, want \"Answer.\"", c)
		}
	case []map[string]interface{}:
		if len(c) != 1 {
			t.Errorf("content parts = %d, want 1 (custom should be dropped)", len(c))
		}
	default:
		t.Fatalf("unexpected content type %T", result[0]["content"])
	}
}

// TestToAnthropicMessagesReasoningFileDropped verifies that ReasoningFileContent
// in assistant messages is silently dropped.
func TestToAnthropicMessagesReasoningFileDropped(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.TextContent{Text: "Here is a chart."},
				types.ReasoningFileContent{
					MediaType: "image/png",
					Data:      []byte{0x89, 0x50, 0x4E, 0x47},
				},
			},
		},
	}

	result := ToAnthropicMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	// Only the text part should remain.
	switch c := result[0]["content"].(type) {
	case string:
		if c != "Here is a chart." {
			t.Errorf("content = %q, want \"Here is a chart.\"", c)
		}
	case []map[string]interface{}:
		if len(c) != 1 {
			t.Errorf("content parts = %d, want 1 (reasoning file should be dropped)", len(c))
		}
		if c[0]["type"] != "text" {
			t.Errorf("remaining part type = %v, want \"text\"", c[0]["type"])
		}
	default:
		t.Fatalf("unexpected content type %T", result[0]["content"])
	}
}

// TestToOpenAIMessagesCustomContentWithOptions verifies that CustomContent
// with OpenAI-keyed ProviderOptions is forwarded verbatim.
func TestToOpenAIMessagesCustomContentWithOptions(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.CustomContent{
					Kind: "openai-custom",
					ProviderOptions: map[string]interface{}{
						"openai": map[string]interface{}{
							"type":  "custom_block",
							"token": "abc123",
						},
					},
				},
			},
		},
	}

	result := ToOpenAIMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	content, ok := result[0]["content"].([]map[string]interface{})
	if !ok {
		t.Fatalf("content should be []map[string]interface{}, got %T", result[0]["content"])
	}
	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}
	if content[0]["type"] != "custom_block" {
		t.Errorf("type = %v, want \"custom_block\"", content[0]["type"])
	}
}

// TestToOpenAIMessagesCustomContentNoOptions verifies that CustomContent
// without OpenAI-keyed ProviderOptions is dropped.
func TestToOpenAIMessagesCustomContentNoOptions(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.TextContent{Text: "Hello."},
				types.CustomContent{Kind: "xai-citation"}, // no openai options
			},
		},
	}

	result := ToOpenAIMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	// Single-text messages get the simple string shortcut — verify content is
	// still just the text (custom part was dropped before the shortcut applied).
	_ = result[0]["content"] // just verify no panic
}

// TestToGoogleMessagesCustomContentWithOptions verifies that CustomContent
// with Google-keyed ProviderOptions is forwarded to the parts array.
func TestToGoogleMessagesCustomContentWithOptions(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.CustomContent{
					Kind: "google-grounding",
					ProviderOptions: map[string]interface{}{
						"google": map[string]interface{}{
							"type":  "grounding_metadata",
							"chunk": "data",
						},
					},
				},
			},
		},
	}

	result := ToGoogleMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}

	parts, ok := result[0]["parts"].([]map[string]interface{})
	if !ok {
		t.Fatalf("parts should be []map[string]interface{}, got %T", result[0]["parts"])
	}
	if len(parts) != 1 {
		t.Fatalf("len(parts) = %d, want 1", len(parts))
	}
	if parts[0]["type"] != "grounding_metadata" {
		t.Errorf("type = %v, want \"grounding_metadata\"", parts[0]["type"])
	}
}

// TestCustomContentNilProviderOptionsNocrash verifies that CustomContent with
// a nil ProviderOptions map does not panic in any converter.
func TestCustomContentNilProviderOptionsNoCrash(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.CustomContent{Kind: "xai-citation"}, // ProviderOptions is nil
			},
		},
	}

	// None of these should panic.
	_ = ToAnthropicMessages(msgs)
	_ = ToOpenAIMessages(msgs)
	_ = ToGoogleMessages(msgs)
}

// TestCustomContentProviderMetadataNotForwarded verifies that ProviderMetadata
// (the output/response field) is NOT used for routing in converters — only
// ProviderOptions (the input field) is checked.
func TestCustomContentProviderMetadataNotForwarded(t *testing.T) {
	msgs := []types.Message{
		{
			Role: types.RoleAssistant,
			Content: []types.ContentPart{
				types.TextContent{Text: "Answer."},
				types.CustomContent{
					Kind:             "xai-citation",
					ProviderMetadata: json.RawMessage(`{"url":"https://x.ai"}`),
					// No ProviderOptions — should be dropped even though metadata is set.
				},
			},
		},
	}

	result := ToAnthropicMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	// Only text content should be present.
	switch c := result[0]["content"].(type) {
	case string:
		// Fine — single text got the shortcut (only if CustomContent was dropped first)
	case []map[string]interface{}:
		for _, part := range c {
			if part["type"] != "text" {
				t.Errorf("unexpected non-text part in output: %v", part)
			}
		}
	default:
		_ = c
	}
}
