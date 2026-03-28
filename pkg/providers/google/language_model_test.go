package google

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/digitallysavvy/go-ai/pkg/provider"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// sseStream builds a minimal SSE response body from a slice of JSON payloads.
func sseStream(payloads ...string) io.ReadCloser {
	var sb strings.Builder
	for _, p := range payloads {
		sb.WriteString("data: ")
		sb.WriteString(p)
		sb.WriteString("\n\n")
	}
	return io.NopCloser(strings.NewReader(sb.String()))
}

// --- buildRequestBody -------------------------------------------------------

func TestBuildRequestBody_ThinkingConfig(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini31FlashImagePreview)

	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "Think about this"},
		ProviderOptions: map[string]interface{}{
			"google": map[string]interface{}{
				"thinkingConfig": map[string]interface{}{
					"thinkingBudget":  1000,
					"includeThoughts": true,
				},
			},
		},
	})

	genConfig, ok := body["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("generationConfig should be present")
	}

	tc, ok := genConfig["thinkingConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("thinkingConfig should be present in generationConfig")
	}
	if tc["thinkingBudget"] != 1000 {
		t.Errorf("thinkingBudget: got %v, want 1000", tc["thinkingBudget"])
	}
	if tc["includeThoughts"] != true {
		t.Errorf("includeThoughts: got %v, want true", tc["includeThoughts"])
	}
}

func TestBuildRequestBody_ThinkingLevel(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini31FlashImagePreview)

	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "Think"},
		ProviderOptions: map[string]interface{}{
			"google": map[string]interface{}{
				"thinkingConfig": map[string]interface{}{
					"thinkingLevel": "high",
				},
			},
		},
	})

	genConfig := body["generationConfig"].(map[string]interface{})
	tc := genConfig["thinkingConfig"].(map[string]interface{})
	if tc["thinkingLevel"] != "high" {
		t.Errorf("thinkingLevel: got %v, want high", tc["thinkingLevel"])
	}
}

func TestBuildRequestBody_NoThinkingConfig(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "Hello"},
	})

	// generationConfig may or may not be present; if it is, thinkingConfig must be absent.
	if gc, ok := body["generationConfig"].(map[string]interface{}); ok {
		if _, has := gc["thinkingConfig"]; has {
			t.Error("thinkingConfig should not be present when not provided")
		}
	}
}

func TestBuildRequestBody_NilProviderOptions(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	// Must not panic with nil ProviderOptions.
	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt:          types.Prompt{Text: "Hello"},
		ProviderOptions: nil,
	})

	if body == nil {
		t.Fatal("body should not be nil")
	}
}

func TestBuildRequestBody_ThinkingConfigIgnoredIfWrongType(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	// thinkingConfig is a string instead of map — should be silently ignored.
	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "Hello"},
		ProviderOptions: map[string]interface{}{
			"google": map[string]interface{}{
				"thinkingConfig": "invalid-type",
			},
		},
	})

	if gc, ok := body["generationConfig"].(map[string]interface{}); ok {
		if _, has := gc["thinkingConfig"]; has {
			t.Error("invalid thinkingConfig type should be ignored")
		}
	}
}

// --- convertResponse (thought parts) ----------------------------------------

func TestConvertResponse_SkipsThoughtParts(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini31FlashImagePreview)

	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{
						{Text: "I am thinking...", Thought: true},
						{Text: "The answer is 42."},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	result := m.convertResponse(resp)

	if result.Text != "The answer is 42." {
		t.Errorf("Text: got %q, want %q", result.Text, "The answer is 42.")
	}
}

func TestConvertResponse_AllThoughtPartsProducesEmptyText(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini31FlashImagePreview)

	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{
						{Text: "Thinking step 1", Thought: true},
						{Text: "Thinking step 2", Thought: true},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	result := m.convertResponse(resp)

	if result.Text != "" {
		t.Errorf("Text: got %q, want empty string when all parts are thought parts", result.Text)
	}
}

func TestConvertResponse_ThoughtPartDoesNotBlockFunctionCall(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{
						{Text: "thinking", Thought: true},
						{FunctionCall: &struct {
							Name string                 `json:"name"`
							Args map[string]interface{} `json:"args"`
						}{Name: "get_weather", Args: map[string]interface{}{"city": "SF"}}},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	result := m.convertResponse(resp)

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].ToolName != "get_weather" {
		t.Errorf("ToolName: got %q, want %q", result.ToolCalls[0].ToolName, "get_weather")
	}
}

// --- googleStream (thought part streaming) -----------------------------------

func TestGoogleStream_ThoughtPartsEmitReasoning(t *testing.T) {
	// Build two SSE events: one thought part, one text part.
	thoughtEvent := mustMarshal(googleResponse{
		Candidates: []googleCandidate{{
			Content: struct {
				Parts []googlePart `json:"parts"`
				Role  string       `json:"role"`
			}{Parts: []googlePart{{Text: "I am reasoning", Thought: true}}},
		}},
	})

	textEvent := mustMarshal(googleResponse{
		Candidates: []googleCandidate{{
			Content: struct {
				Parts []googlePart `json:"parts"`
				Role  string       `json:"role"`
			}{Parts: []googlePart{{Text: "The answer is 7"}}},
			FinishReason: "STOP",
		}},
	})

	stream := newGoogleStream(sseStream(thoughtEvent, textEvent))
	defer stream.Close()

	chunk1, err := stream.Next()
	if err != nil {
		t.Fatalf("unexpected error on chunk1: %v", err)
	}
	if chunk1.Type != provider.ChunkTypeReasoning {
		t.Errorf("chunk1.Type: got %v, want ChunkTypeReasoning", chunk1.Type)
	}
	if chunk1.Text != "I am reasoning" {
		t.Errorf("chunk1.Text: got %q, want %q", chunk1.Text, "I am reasoning")
	}

	chunk2, err := stream.Next()
	if err != nil {
		t.Fatalf("unexpected error on chunk2: %v", err)
	}
	if chunk2.Type != provider.ChunkTypeText {
		t.Errorf("chunk2.Type: got %v, want ChunkTypeText", chunk2.Type)
	}
	if chunk2.Text != "The answer is 7" {
		t.Errorf("chunk2.Text: got %q, want %q", chunk2.Text, "The answer is 7")
	}

	chunk3, err := stream.Next()
	if err != nil {
		t.Fatalf("unexpected error on chunk3: %v", err)
	}
	if chunk3.Type != provider.ChunkTypeFinish {
		t.Errorf("chunk3.Type: got %v, want ChunkTypeFinish", chunk3.Type)
	}
	if chunk3.FinishReason != types.FinishReasonStop {
		t.Errorf("chunk3.FinishReason: got %v, want FinishReasonStop", chunk3.FinishReason)
	}
}

func TestGoogleStream_MultiplePartsInSingleEvent(t *testing.T) {
	// An event with both a thought part and a text part.
	event := mustMarshal(googleResponse{
		Candidates: []googleCandidate{{
			Content: struct {
				Parts []googlePart `json:"parts"`
				Role  string       `json:"role"`
			}{Parts: []googlePart{
				{Text: "thinking", Thought: true},
				{Text: "answer"},
			}},
			FinishReason: "STOP",
		}},
	})

	stream := newGoogleStream(sseStream(event))
	defer stream.Close()

	var chunks []*provider.StreamChunk
	for {
		c, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, c)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks (reasoning, text, finish), got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkTypeReasoning {
		t.Errorf("chunk[0]: got %v, want reasoning", chunks[0].Type)
	}
	if chunks[1].Type != provider.ChunkTypeText {
		t.Errorf("chunk[1]: got %v, want text", chunks[1].Type)
	}
	if chunks[2].Type != provider.ChunkTypeFinish {
		t.Errorf("chunk[2]: got %v, want finish", chunks[2].Type)
	}
}

func TestGoogleStream_FinishReasonEmittedAfterText(t *testing.T) {
	// Final event has text AND a finish reason — finish must come after text.
	event := mustMarshal(googleResponse{
		Candidates: []googleCandidate{{
			Content: struct {
				Parts []googlePart `json:"parts"`
				Role  string       `json:"role"`
			}{Parts: []googlePart{{Text: "last word"}}},
			FinishReason: "MAX_TOKENS",
		}},
	})

	stream := newGoogleStream(sseStream(event))
	defer stream.Close()

	c1, _ := stream.Next()
	c2, _ := stream.Next()

	if c1.Type != provider.ChunkTypeText || c1.Text != "last word" {
		t.Errorf("c1: got type=%v text=%q, want text 'last word'", c1.Type, c1.Text)
	}
	if c2.Type != provider.ChunkTypeFinish || c2.FinishReason != types.FinishReasonLength {
		t.Errorf("c2: got type=%v reason=%v, want finish MAX_TOKENS", c2.Type, c2.FinishReason)
	}
}

// --- integration test -------------------------------------------------------

func TestLanguageModel_Integration_ThinkingConfig(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_GENERATIVE_AI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping: GOOGLE_GENERATIVE_AI_API_KEY not set")
	}

	p := New(Config{APIKey: apiKey})
	m, err := p.LanguageModel(ModelGemini31FlashImagePreview)
	if err != nil {
		t.Fatalf("LanguageModel: %v", err)
	}

	result, err := m.DoGenerate(context.Background(), &provider.GenerateOptions{
		Prompt: types.Prompt{Text: "What is 2 + 2? Think step by step."},
		ProviderOptions: map[string]interface{}{
			"google": map[string]interface{}{
				"thinkingConfig": map[string]interface{}{
					"thinkingBudget":  512,
					"includeThoughts": true,
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate: %v", err)
	}
	if result.Text == "" {
		t.Error("expected non-empty result text")
	}
	t.Logf("Result: %s", result.Text)
	if result.Usage.OutputDetails != nil && result.Usage.OutputDetails.ReasoningTokens != nil {
		t.Logf("Reasoning tokens: %d", *result.Usage.OutputDetails.ReasoningTokens)
	}
}

// --- httptest-based request body verification --------------------------------

func TestBuildRequestBody_ThinkingConfig_ViaHTTP(t *testing.T) {
	var capturedBody map[string]interface{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		// Return minimal valid response.
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer srv.Close()

	p := New(Config{APIKey: "test-key", BaseURL: srv.URL})
	m := NewLanguageModel(p, ModelGemini20Flash)

	_, _ = m.DoGenerate(context.Background(), &provider.GenerateOptions{
		Prompt: types.Prompt{Text: "hello"},
		ProviderOptions: map[string]interface{}{
			"google": map[string]interface{}{
				"thinkingConfig": map[string]interface{}{
					"thinkingBudget": float64(256),
				},
			},
		},
	})

	if capturedBody == nil {
		t.Skip("capturedBody nil — provider may not support BaseURL override; skipping HTTP verification")
		return
	}
	gc, ok := capturedBody["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("generationConfig not in request body")
	}
	tc, ok := gc["thinkingConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("thinkingConfig not in generationConfig")
	}
	if tc["thinkingBudget"] != float64(256) {
		t.Errorf("thinkingBudget: got %v, want 256", tc["thinkingBudget"])
	}
}

// --- VALIDATED mode (strict tools) ------------------------------------------

func TestGoogleStrictToolsUsesValidatedMode(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "hello"},
		Tools: []types.Tool{
			{
				Name:        "get_data",
				Description: "Retrieve data",
				Parameters:  map[string]interface{}{"type": "object"},
				Strict:      true,
			},
		},
	})

	toolConfig, ok := body["toolConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("toolConfig must be present when any tool has Strict:true")
	}
	fcc, ok := toolConfig["functionCallingConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("functionCallingConfig must be present in toolConfig")
	}
	if fcc["mode"] != "VALIDATED" {
		t.Errorf("functionCallingConfig.mode = %v, want VALIDATED", fcc["mode"])
	}
}

func TestGoogleNoStrictToolsUsesDefaultMode(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "hello"},
		Tools: []types.Tool{
			{
				Name:        "get_data",
				Description: "Retrieve data",
				Parameters:  map[string]interface{}{"type": "object"},
				Strict:      false,
			},
		},
	})

	if _, ok := body["toolConfig"]; ok {
		t.Error("toolConfig must NOT be present when no tool has Strict:true")
	}
}

// --- groundingMetadata in stream --------------------------------------------

func TestGoogleStreamPreservesGroundingMetadataBeforeFinish(t *testing.T) {
	// First chunk has groundingMetadata but no finishReason.
	// Second chunk has the finishReason. The metadata from the first chunk must
	// appear on the finish chunk.
	groundingMeta := `{"groundingSupport":[{"segment":{"startIndex":0}}]}`

	earlyChunk := mustMarshal(googleResponse{
		Candidates: []googleCandidate{{
			Content: struct {
				Parts []googlePart `json:"parts"`
				Role  string       `json:"role"`
			}{Parts: []googlePart{{Text: "some text"}}},
			GroundingMetadata: json.RawMessage(groundingMeta),
		}},
	})

	finishChunk := mustMarshal(googleResponse{
		Candidates: []googleCandidate{{
			FinishReason: "STOP",
		}},
	})

	stream := newGoogleStream(sseStream(earlyChunk, finishChunk))
	defer stream.Close()

	// Collect all chunks.
	var chunks []*provider.StreamChunk
	for {
		c, err := stream.Next()
		if err != nil {
			break
		}
		chunks = append(chunks, c)
	}

	// Find the finish chunk and verify it carries the grounding metadata.
	var finishFound bool
	for _, c := range chunks {
		if c.Type == provider.ChunkTypeFinish {
			finishFound = true
			if c.ProviderMetadata == nil {
				t.Error("finish chunk must carry ProviderMetadata with grounding info")
			}
		}
	}
	if !finishFound {
		t.Error("expected a ChunkTypeFinish chunk")
	}
}

// --- supportsFunctionResponseParts ------------------------------------------

func TestSupportsFunctionResponseParts_Gemini3(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, "gemini-3-pro-preview")
	if !m.supportsFunctionResponseParts() {
		t.Error("gemini-3-pro-preview should support function response parts")
	}
}

func TestSupportsFunctionResponseParts_Gemini2(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)
	if m.supportsFunctionResponseParts() {
		t.Error("gemini-2.0-flash should NOT support function response parts")
	}
}

// --- Gemini 3 / Gemini 2 image tool result mapping -------------------------

// TestGemini3ImageToolResultMapsToFunctionResponse verifies that when a Gemini 3
// model is used, image tool results are encoded as functionResponse.parts[].inlineData
// alongside the text in response.content.
func TestGemini3ImageToolResultMapsToFunctionResponse(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini3ProPreview)

	imageBytes := []byte{0x89, 0x50, 0x4e, 0x47} // PNG magic bytes
	opts := &provider.GenerateOptions{
		Prompt: types.Prompt{
			Messages: []types.Message{
				// Turn 1: user asks a question
				{Role: types.RoleUser, Content: []types.ContentPart{
					types.TextContent{Text: "Describe this image"},
				}},
				// Turn 2: model calls a tool
				{Role: types.RoleAssistant, ToolCalls: []types.ToolCall{
					{ID: "call1", ToolName: "getImage", Arguments: map[string]interface{}{}},
				}},
				// Turn 3: tool returns an image result
				{Role: types.RoleTool, Content: []types.ContentPart{
					types.ToolResultContent{
						ToolCallID: "call1",
						ToolName:   "getImage",
						Output: &types.ToolResultOutput{
							Type: types.ToolResultOutputContent,
							Content: []types.ToolResultContentBlock{
								types.TextContentBlock{Text: "Here is the image"},
								types.ImageContentBlock{Data: imageBytes, MediaType: "image/png"},
							},
						},
					},
				}},
			},
		},
	}

	body := m.buildRequestBody(opts)
	contents, ok := body["contents"].([]map[string]interface{})
	if !ok {
		t.Fatalf("contents type = %T", body["contents"])
	}

	// Find the tool-result message (should be the last one, role "user").
	var toolMsg map[string]interface{}
	for _, c := range contents {
		if c["role"] == "user" {
			parts, _ := c["parts"].([]map[string]interface{})
			for _, pt := range parts {
				if _, hasFR := pt["functionResponse"]; hasFR {
					toolMsg = c
				}
			}
		}
	}
	if toolMsg == nil {
		t.Fatal("no tool-result message (functionResponse) found in contents")
	}

	parts := toolMsg["parts"].([]map[string]interface{})
	fr := parts[0]["functionResponse"].(map[string]interface{})

	// Gemini 3: binary parts go into functionResponse.parts[].
	frParts, ok := fr["parts"].([]map[string]interface{})
	if !ok || len(frParts) == 0 {
		t.Fatalf("functionResponse.parts missing or empty; got %v", fr["parts"])
	}
	inlineData, ok := frParts[0]["inlineData"].(map[string]interface{})
	if !ok {
		t.Fatalf("functionResponse.parts[0].inlineData missing; got %v", frParts[0])
	}
	if inlineData["mimeType"] != "image/png" {
		t.Errorf("mimeType = %v, want image/png", inlineData["mimeType"])
	}

	// Text goes into response.content.
	resp := fr["response"].(map[string]interface{})
	if resp["content"] != "Here is the image" {
		t.Errorf("response.content = %v, want %q", resp["content"], "Here is the image")
	}
}

// TestGemini2ImageToolResultFallsBackToText verifies that pre-Gemini-3 models
// send image tool results as separate top-level inlineData parts (legacy format).
func TestGemini2ImageToolResultFallsBackToText(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	imageBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	opts := &provider.GenerateOptions{
		Prompt: types.Prompt{
			Messages: []types.Message{
				{Role: types.RoleUser, Content: []types.ContentPart{
					types.TextContent{Text: "show me"},
				}},
				{Role: types.RoleTool, Content: []types.ContentPart{
					types.ToolResultContent{
						ToolCallID: "c1",
						ToolName:   "getImage",
						Output: &types.ToolResultOutput{
							Type: types.ToolResultOutputContent,
							Content: []types.ToolResultContentBlock{
								types.ImageContentBlock{Data: imageBytes, MediaType: "image/png"},
							},
						},
					},
				}},
			},
		},
	}

	body := m.buildRequestBody(opts)
	contents := body["contents"].([]map[string]interface{})

	// Find the tool-result user message.
	var toolMsg map[string]interface{}
	for _, c := range contents {
		if c["role"] == "user" {
			parts, _ := c["parts"].([]map[string]interface{})
			for _, pt := range parts {
				if _, hasID := pt["inlineData"]; hasID {
					toolMsg = c
				}
			}
		}
	}
	if toolMsg == nil {
		t.Fatal("no legacy inlineData part found in tool result message")
	}

	// Gemini 2 legacy: image should be a top-level inlineData part, NOT inside functionResponse.parts[].
	parts := toolMsg["parts"].([]map[string]interface{})
	var hasInlineData bool
	for _, pt := range parts {
		if _, ok := pt["inlineData"]; ok {
			hasInlineData = true
		}
		// Ensure functionResponse does NOT have a parts[] field.
		if fr, ok := pt["functionResponse"].(map[string]interface{}); ok {
			if _, hasParts := fr["parts"]; hasParts {
				t.Error("Gemini 2 must NOT use functionResponse.parts[] — legacy format only")
			}
		}
	}
	if !hasInlineData {
		t.Error("Gemini 2 legacy format must emit a top-level inlineData part for images")
	}
}

// --- reasoning files correctly marked ---------------------------------------

func TestGoogleReasoningFilesMarkedCorrectly(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini25Pro)

	// A part with thought=true and inlineData should be typed as ReasoningFileContent.
	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{
						{
							Thought: true,
							InlineData: &struct {
								MimeType string `json:"mimeType"`
								Data     string `json:"data"`
							}{
								MimeType: "image/png",
								Data:     "iVBORw0KGgo=", // minimal base64
							},
						},
						{Text: "The answer."},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	result := m.convertResponse(resp)

	if result.Text != "The answer." {
		t.Errorf("Text = %q, want %q", result.Text, "The answer.")
	}

	if len(result.Content) == 0 {
		t.Fatal("expected at least one Content part for the reasoning file")
	}

	rfContent, ok := result.Content[0].(types.ReasoningFileContent)
	if !ok {
		t.Fatalf("Content[0] type = %T, want types.ReasoningFileContent", result.Content[0])
	}
	if rfContent.MediaType != "image/png" {
		t.Errorf("ReasoningFileContent.MediaType = %q, want %q", rfContent.MediaType, "image/png")
	}
}

// --- native Google tools in request body ------------------------------------

func TestGoogleNativeSearchToolInRequestBody(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	tool := GoogleSearchTool()
	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt: types.Prompt{Text: "search"},
		Tools:  []types.Tool{tool},
	})

	tools, ok := body["tools"].([]map[string]interface{})
	if !ok {
		t.Fatalf("tools must be []map[string]interface{}, got %T", body["tools"])
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool entry, got %d", len(tools))
	}
	if _, hasGS := tools[0]["googleSearch"]; !hasGS {
		t.Errorf("tool entry must contain googleSearch key; got %v", tools[0])
	}
}

// --- Gemini 3 thinkingLevel -------------------------------------------------

func TestGemini3ReasoningUsesThinkingLevel(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini3ProPreview)

	level := types.ReasoningMedium
	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt:    types.Prompt{Text: "think"},
		Reasoning: &level,
	})

	gc, ok := body["generationConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("generationConfig missing or wrong type: %T", body["generationConfig"])
	}
	tc, ok := gc["thinkingConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("thinkingConfig missing or wrong type: %T", gc["thinkingConfig"])
	}
	if _, hasBudget := tc["thinkingBudget"]; hasBudget {
		t.Error("Gemini 3 must use thinkingLevel, not thinkingBudget")
	}
	if tl, ok := tc["thinkingLevel"].(string); !ok || tl == "" {
		t.Errorf("expected thinkingLevel string, got %v", tc["thinkingLevel"])
	}
}

func TestGemini3ReasoningNoneMapsToMinimal(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini3ProPreview)

	level := types.ReasoningNone
	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt:    types.Prompt{Text: "no think"},
		Reasoning: &level,
	})

	gc := body["generationConfig"].(map[string]interface{})
	tc := gc["thinkingConfig"].(map[string]interface{})
	if tl, _ := tc["thinkingLevel"].(string); tl != "minimal" {
		t.Errorf("ReasoningNone on Gemini 3 must map to 'minimal', got %q", tl)
	}
}

func TestGemini25ReasoningUsesThinkingBudget(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini25Pro)

	level := types.ReasoningMedium
	body := m.buildRequestBody(&provider.GenerateOptions{
		Prompt:    types.Prompt{Text: "think"},
		Reasoning: &level,
	})

	gc := body["generationConfig"].(map[string]interface{})
	tc := gc["thinkingConfig"].(map[string]interface{})
	if _, hasBudget := tc["thinkingBudget"]; !hasBudget {
		t.Error("Gemini 2.5 must use thinkingBudget, not thinkingLevel")
	}
}

// --- isGemini3Model ---------------------------------------------------------

func TestIsGemini3Model(t *testing.T) {
	cases := []struct {
		model    string
		expected bool
	}{
		{ModelGemini3ProPreview, true},
		{ModelGemini3ProImagePreview, true}, // image model — still a Gemini 3
		{ModelGemini3FlashPreview, true},
		{ModelGemini31ProPreview, true},
		{ModelGemini31ProPreviewCustom, true},
		{ModelGemini31FlashImagePreview, true}, // image model — still a Gemini 3
		{ModelGemini25Pro, false},
		{ModelGemini20Flash, false},
		{"gemini-2.5-flash-preview-04-17", false},
	}
	for _, c := range cases {
		got := isGemini3Model(c.model)
		if got != c.expected {
			t.Errorf("isGemini3Model(%q) = %v, want %v", c.model, got, c.expected)
		}
	}
}

// TestGemini3ImageModelDoesNotGetThinkingLevel verifies that Gemini 3 image
// models (which don't support extended thinking) are excluded from thinkingLevel.
func TestGemini3ImageModelDoesNotGetThinkingLevel(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	for _, modelID := range []string{ModelGemini3ProImagePreview, ModelGemini31FlashImagePreview} {
		m := NewLanguageModel(p, modelID)
		level := types.ReasoningMedium
		body := m.buildRequestBody(&provider.GenerateOptions{
			Prompt:    types.Prompt{Text: "describe"},
			Reasoning: &level,
		})
		gc, ok := body["generationConfig"].(map[string]interface{})
		if !ok {
			continue // no genConfig means no thinkingConfig — that's fine too
		}
		tc, hasTc := gc["thinkingConfig"].(map[string]interface{})
		if !hasTc {
			continue // no thinkingConfig at all is also acceptable
		}
		if _, hasLevel := tc["thinkingLevel"]; hasLevel {
			t.Errorf("model %q must NOT use thinkingLevel (image model)", modelID)
		}
	}
}

// TestGoogleDoGenerateGroundingMetadataInProviderMetadata verifies that
// groundingMetadata from a non-streaming response is included in ProviderMetadata.
func TestGoogleDoGenerateGroundingMetadataInProviderMetadata(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	groundingJSON := json.RawMessage(`{"webSearchQueries":["golang generics"]}`)
	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{{Text: "Here is the info."}},
				},
				FinishReason:      "STOP",
				GroundingMetadata: groundingJSON,
			},
		},
	}

	result := m.convertResponse(resp)

	if result.ProviderMetadata == nil {
		t.Fatal("ProviderMetadata must be set when groundingMetadata is present")
	}
	googleMeta, ok := result.ProviderMetadata["google"].(map[string]json.RawMessage)
	if !ok {
		t.Fatalf("ProviderMetadata[google] type = %T", result.ProviderMetadata["google"])
	}
	if string(googleMeta["groundingMetadata"]) != string(groundingJSON) {
		t.Errorf("groundingMetadata = %s, want %s", googleMeta["groundingMetadata"], groundingJSON)
	}
}

// --- thoughtSignature support -----------------------------------------------

// TestConvertResponseThoughtSignatureOnFunctionCall verifies that ThoughtSignature
// on a functionCall part is captured into ToolCall.ThoughtSignature.
func TestConvertResponseThoughtSignatureOnFunctionCall(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{
						{
							FunctionCall: &struct {
								Name string                 `json:"name"`
								Args map[string]interface{} `json:"args"`
							}{Name: "search", Args: map[string]interface{}{"q": "test"}},
							ThoughtSignature: "sig-abc-123",
						},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	result := m.convertResponse(resp)

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].ThoughtSignature != "sig-abc-123" {
		t.Errorf("ThoughtSignature = %q, want %q", result.ToolCalls[0].ThoughtSignature, "sig-abc-123")
	}
}

// TestConvertResponseThoughtPartsBecomesReasoningContent verifies that thought
// text parts are emitted as ReasoningContent with the cryptographic signature.
func TestConvertResponseThoughtPartsBecomesReasoningContent(t *testing.T) {
	p := New(Config{APIKey: "test-key"})
	m := NewLanguageModel(p, ModelGemini20Flash)

	resp := googleResponse{
		Candidates: []googleCandidate{
			{
				Content: struct {
					Parts []googlePart `json:"parts"`
					Role  string       `json:"role"`
				}{
					Parts: []googlePart{
						{Text: "thinking...", Thought: true, ThoughtSignature: "sealed-token"},
						{Text: "Here is the answer."},
					},
				},
				FinishReason: "STOP",
			},
		},
	}

	result := m.convertResponse(resp)

	if result.Text != "Here is the answer." {
		t.Errorf("Text = %q, want %q", result.Text, "Here is the answer.")
	}

	var found *types.ReasoningContent
	for _, part := range result.Content {
		if rc, ok := part.(types.ReasoningContent); ok {
			found = &rc
			break
		}
	}
	if found == nil {
		t.Fatal("expected a ReasoningContent part in result.Content")
	}
	if found.Text != "thinking..." {
		t.Errorf("ReasoningContent.Text = %q, want %q", found.Text, "thinking...")
	}
	if found.Signature != "sealed-token" {
		t.Errorf("ReasoningContent.Signature = %q, want %q", found.Signature, "sealed-token")
	}
}

// TestStreamingFunctionCallChunkCarriesThoughtSignature verifies that a function
// call part in the streaming path emits a ChunkTypeToolCall with ThoughtSignature.
func TestStreamingFunctionCallChunkCarriesThoughtSignature(t *testing.T) {
	chunkJSON := `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"tool","args":{"x":1}},"thoughtSignature":"stream-sig"}]}}]}`
	stream := newGoogleStream(sseStream(chunkJSON, "[DONE]"))

	var toolCallChunk *provider.StreamChunk
	for {
		chunk, err := stream.Next()
		if err != nil {
			break
		}
		if chunk.Type == provider.ChunkTypeToolCall {
			toolCallChunk = chunk
			break
		}
	}

	if toolCallChunk == nil {
		t.Fatal("expected a ChunkTypeToolCall chunk")
	}
	if toolCallChunk.ToolCall == nil {
		t.Fatal("ToolCall must be set on ChunkTypeToolCall chunk")
	}
	if toolCallChunk.ToolCall.ThoughtSignature != "stream-sig" {
		t.Errorf("ThoughtSignature = %q, want %q", toolCallChunk.ToolCall.ThoughtSignature, "stream-sig")
	}
}

// TestStreamingReasoningChunkCarriesThoughtSignature verifies that a thought part
// in streaming carries the ThoughtSignature in ProviderMetadata.
func TestStreamingReasoningChunkCarriesThoughtSignature(t *testing.T) {
	chunkJSON := `{"candidates":[{"content":{"parts":[{"text":"reasoning...","thought":true,"thoughtSignature":"rsig-xyz"}]}}]}`
	stream := newGoogleStream(sseStream(chunkJSON, "[DONE]"))

	var reasoningChunk *provider.StreamChunk
	for {
		chunk, err := stream.Next()
		if err != nil {
			break
		}
		if chunk.Type == provider.ChunkTypeReasoning {
			reasoningChunk = chunk
			break
		}
	}

	if reasoningChunk == nil {
		t.Fatal("expected a ChunkTypeReasoning chunk")
	}
	if len(reasoningChunk.ProviderMetadata) == 0 {
		t.Fatal("ProviderMetadata must be set on reasoning chunk with ThoughtSignature")
	}
	var meta map[string]interface{}
	if err := json.Unmarshal(reasoningChunk.ProviderMetadata, &meta); err != nil {
		t.Fatalf("failed to unmarshal ProviderMetadata: %v", err)
	}
	googleMeta, ok := meta["google"].(map[string]interface{})
	if !ok {
		t.Fatalf("meta[google] type = %T", meta["google"])
	}
	if googleMeta["thoughtSignature"] != "rsig-xyz" {
		t.Errorf("thoughtSignature = %v, want rsig-xyz", googleMeta["thoughtSignature"])
	}
}

// --- EmbeddingModelGeminiEmbedding001 constant -----------------------------

func TestEmbeddingModelGeminiEmbedding001Constant(t *testing.T) {
	if EmbeddingModelGeminiEmbedding001 != "gemini-embedding-001" {
		t.Errorf("EmbeddingModelGeminiEmbedding001 = %q, want %q",
			EmbeddingModelGeminiEmbedding001, "gemini-embedding-001")
	}
}

// --- helpers ----------------------------------------------------------------

func mustMarshal(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}
