package googlevertex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	internalhttp "github.com/digitallysavvy/go-ai/pkg/internal/http"
	"github.com/digitallysavvy/go-ai/pkg/provider"
	providererrors "github.com/digitallysavvy/go-ai/pkg/provider/errors"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
	"github.com/digitallysavvy/go-ai/pkg/providerutils/prompt"
	"github.com/digitallysavvy/go-ai/pkg/providerutils/streaming"
	"github.com/digitallysavvy/go-ai/pkg/providerutils/tool"
)

// LanguageModel implements the provider.LanguageModel interface for Google Vertex AI
// It uses the same Gemini models and API format as Google Generative AI
// but with Vertex AI endpoints and authentication
type LanguageModel struct {
	provider *Provider
	modelID  string
}

// NewLanguageModel creates a new Google Vertex AI language model
func NewLanguageModel(provider *Provider, modelID string) *LanguageModel {
	return &LanguageModel{
		provider: provider,
		modelID:  modelID,
	}
}

// SpecificationVersion returns the specification version
func (m *LanguageModel) SpecificationVersion() string {
	return "v3"
}

// Provider returns the provider name
func (m *LanguageModel) Provider() string {
	return "google-vertex"
}

// ModelID returns the model ID
func (m *LanguageModel) ModelID() string {
	return m.modelID
}

// SupportsTools returns whether the model supports tool calling
func (m *LanguageModel) SupportsTools() bool {
	// Gemini Pro models support function calling
	return true
}

// SupportsStructuredOutput returns whether the model supports structured output
func (m *LanguageModel) SupportsStructuredOutput() bool {
	// Gemini supports JSON mode
	return true
}

// SupportsImageInput returns whether the model accepts image inputs
func (m *LanguageModel) SupportsImageInput() bool {
	// Gemini Pro Vision and 1.5 models support images
	return m.modelID == "gemini-pro-vision" ||
		m.modelID == "gemini-1.5-pro" ||
		m.modelID == "gemini-1.5-flash" ||
		m.modelID == "gemini-1.5-flash-8b" ||
		m.modelID == "gemini-2.0-flash-exp"
}

// DoGenerate performs non-streaming text generation
func (m *LanguageModel) DoGenerate(ctx context.Context, opts *provider.GenerateOptions) (*types.GenerateResult, error) {
	// Build request body
	reqBody := m.buildRequestBody(opts)

	// Build path for Vertex AI
	// The base URL already includes the project/location/publishers path
	// Format: /models/{model}:generateContent
	path := fmt.Sprintf("/models/%s:generateContent", m.modelID)

	// Make API request
	var response vertexResponse
	err := m.provider.client.PostJSON(ctx, path, reqBody, &response)
	if err != nil {
		return nil, m.handleError(err)
	}

	// Convert response to GenerateResult
	return m.convertResponse(response), nil
}

// DoStream performs streaming text generation
func (m *LanguageModel) DoStream(ctx context.Context, opts *provider.GenerateOptions) (provider.TextStream, error) {
	// Build request body
	reqBody := m.buildRequestBody(opts)

	// Build path for Vertex AI streaming
	// The base URL already includes the project/location/publishers path
	// Format: /models/{model}:streamGenerateContent?alt=sse
	path := fmt.Sprintf("/models/%s:streamGenerateContent?alt=sse", m.modelID)

	// Make streaming API request
	httpResp, err := m.provider.client.DoStream(ctx, internalhttp.Request{
		Method: http.MethodPost,
		Path:   path,
		Body:   reqBody,
		Headers: map[string]string{
			"Accept": "text/event-stream",
		},
	})
	if err != nil {
		return nil, m.handleError(err)
	}

	// Create stream wrapper
	return newVertexStream(httpResp.Body), nil
}

// buildRequestBody builds the Google Vertex AI request body
// Uses the same format as Google Generative AI
func (m *LanguageModel) buildRequestBody(opts *provider.GenerateOptions) map[string]interface{} {
	body := map[string]interface{}{}

	// Convert messages to Google format
	if opts.Prompt.IsMessages() {
		body["contents"] = prompt.ToGoogleMessages(opts.Prompt.Messages, m.supportsFunctionResponseParts())
	} else if opts.Prompt.IsSimple() {
		body["contents"] = prompt.ToGoogleMessages(prompt.SimpleTextToMessages(opts.Prompt.Text), false)
	}

	// Add system instruction if present.
	// Gemma models do not support systemInstruction — matches TS isGemmaModel check.
	if opts.Prompt.System != "" && !isGemmaModel(m.modelID) {
		body["systemInstruction"] = map[string]interface{}{
			"parts": []map[string]interface{}{
				{"text": opts.Prompt.System},
			},
		}
	}

	// Build generation config
	genConfig := map[string]interface{}{}
	if opts.Temperature != nil {
		genConfig["temperature"] = *opts.Temperature
	}
	if opts.MaxTokens != nil {
		genConfig["maxOutputTokens"] = *opts.MaxTokens
	}
	if opts.TopP != nil {
		genConfig["topP"] = *opts.TopP
	}
	if opts.TopK != nil {
		genConfig["topK"] = *opts.TopK
	}
	if len(opts.StopSequences) > 0 {
		genConfig["stopSequences"] = opts.StopSequences
	}
	if opts.FrequencyPenalty != nil {
		genConfig["frequencyPenalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		genConfig["presencePenalty"] = *opts.PresencePenalty
	}
	if opts.Seed != nil {
		genConfig["seed"] = *opts.Seed
	}

	// Map top-level Reasoning to Vertex AI thinkingConfig.
	// Gemini 3 models use thinkingLevel ('minimal'/'low'/'medium'/'high').
	// Gemini 2.x and earlier use thinkingBudget (integer token count).
	// Call-level Reasoning takes precedence over ProviderOptions.
	// provider-default → omit (use model default).
	if opts.Reasoning != nil && *opts.Reasoning != types.ReasoningDefault {
		// Gemini 3 image models do not support extended thinking.
		if isGemini3Model(m.modelID) && !strings.Contains(m.modelID, "image") {
			genConfig["thinkingConfig"] = map[string]interface{}{
				"thinkingLevel": mapReasoningToGemini3Level(*opts.Reasoning),
			}
		} else {
			switch *opts.Reasoning {
			case types.ReasoningNone:
				genConfig["thinkingConfig"] = map[string]interface{}{"thinkingBudget": 0}
			default:
				maxOut := 0
				if opts.MaxTokens != nil {
					maxOut = *opts.MaxTokens
				}
				budget := mapReasoningToVertexBudget(*opts.Reasoning, maxOut, m.modelID)
				genConfig["thinkingConfig"] = map[string]interface{}{"thinkingBudget": budget}
			}
		}
	} else {
		// Fall back to ProviderOptions thinkingConfig when Reasoning is unset.
		// Checks "vertex" key first (TS canonical), then "googleVertex" (legacy Go),
		// then "google" (shared options). Matches TS getArgs() fallback chain.
		if opts.ProviderOptions != nil {
			var thinkingConfig map[string]interface{}
			for _, key := range []string{"vertex", "googleVertex", "google"} {
				if provOpts, ok := opts.ProviderOptions[key].(map[string]interface{}); ok {
					if tc, ok := provOpts["thinkingConfig"].(map[string]interface{}); ok {
						thinkingConfig = tc
						break
					}
				}
			}
			if thinkingConfig != nil {
				genConfig["thinkingConfig"] = thinkingConfig
			}
		}
	}

	// Add response MIME type and schema for JSON mode.
	// Matches TS: responseMimeType when type==='json', responseSchema when schema is present
	// and structuredOutputs !== false (default true).
	if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json_object" {
		genConfig["responseMimeType"] = "application/json"
	}
	if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" {
		genConfig["responseMimeType"] = "application/json"
		if opts.ResponseFormat.Schema != nil {
			// Check if structuredOutputs is explicitly disabled via provider options.
			// Checks "vertex" key first, then "googleVertex"/"google" fallbacks.
			structuredOutputs := true
			if opts.ProviderOptions != nil {
				for _, key := range []string{"vertex", "googleVertex", "google"} {
					if provOpts, ok := opts.ProviderOptions[key].(map[string]interface{}); ok {
						if so, ok := provOpts["structuredOutputs"].(bool); ok {
							structuredOutputs = so
							break
						}
					}
				}
			}
			if structuredOutputs {
				genConfig["responseSchema"] = opts.ResponseFormat.Schema
			}
		}
	}

	// Forward additional provider options from ProviderOptions["vertex"] (or legacy
	// "googleVertex"/"google" fallbacks) into generationConfig and the request body.
	var vertexOpts map[string]interface{}
	if opts.ProviderOptions != nil {
		for _, key := range []string{"vertex", "googleVertex", "google"} {
			if v, ok := opts.ProviderOptions[key].(map[string]interface{}); ok {
				vertexOpts = v
				break
			}
		}
	}
	if vertexOpts != nil {
		for _, key := range []string{"responseModalities", "mediaResolution", "audioTimestamp", "imageConfig"} {
			if v, ok := vertexOpts[key]; ok {
				genConfig[key] = v
			}
		}
		if v, ok := vertexOpts["safetySettings"]; ok {
			body["safetySettings"] = v
		}
		if v, ok := vertexOpts["cachedContent"]; ok {
			body["cachedContent"] = v
		}
		if v, ok := vertexOpts["labels"]; ok {
			body["labels"] = v
		}
	}

	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}

	// Add tools if present — separating provider/native tools from function tools.
	if len(opts.Tools) > 0 {
		var functionTools []types.Tool
		var nativeEntries []map[string]interface{}

		for _, t := range opts.Tools {
			if t.Type == "provider" {
				if entry := buildVertexNativeToolEntry(t); entry != nil {
					nativeEntries = append(nativeEntries, entry)
				}
			} else {
				functionTools = append(functionTools, t)
			}
		}

		if len(nativeEntries) > 0 {
			body["tools"] = nativeEntries
			if vertexOpts != nil {
				if rc, ok := vertexOpts["retrievalConfig"]; ok {
					body["toolConfig"] = map[string]interface{}{"retrievalConfig": rc}
				}
			}
		} else if len(functionTools) > 0 {
			body["tools"] = []map[string]interface{}{
				{"functionDeclarations": tool.ToGoogleFormat(functionTools)},
			}

			// Determine functionCallingConfig mode — matches TS prepareTools.
			hasStrictTools := false
			for _, t := range functionTools {
				if t.Strict {
					hasStrictTools = true
					break
				}
			}

			var mode string
			var allowedFunctionNames []string
			if opts.ToolChoice != (types.ToolChoice{}) {
				switch opts.ToolChoice.Type {
				case types.ToolChoiceNone:
					mode = "NONE"
				case types.ToolChoiceRequired:
					if hasStrictTools {
						mode = "VALIDATED"
					} else {
						mode = "ANY"
					}
				case types.ToolChoiceTool:
					if hasStrictTools {
						mode = "VALIDATED"
					} else {
						mode = "ANY"
					}
					allowedFunctionNames = []string{opts.ToolChoice.ToolName}
				default: // auto
					if hasStrictTools {
						mode = "VALIDATED"
					} else {
						mode = "AUTO"
					}
				}
			} else if hasStrictTools {
				mode = "VALIDATED"
			}

			if mode != "" {
				fcConfig := map[string]interface{}{"mode": mode}
				if len(allowedFunctionNames) > 0 {
					fcConfig["allowedFunctionNames"] = allowedFunctionNames
				}
				body["toolConfig"] = map[string]interface{}{
					"functionCallingConfig": fcConfig,
				}
			}
		}
	}

	return body
}

// maxThinkingTokensForVertexModel returns the maximum thinking token capacity for
// a Google Vertex AI model. Same values as the Google Generative AI provider.
func maxThinkingTokensForVertexModel(modelID string) int {
	switch {
	case strings.Contains(modelID, "gemini-2.5-pro"):
		return 32768
	case strings.Contains(modelID, "gemini-2.5-flash"):
		return 24576
	case strings.Contains(modelID, "gemini-2.0-flash-thinking"):
		return 8192
	default:
		return 8192
	}
}

// mapReasoningToVertexBudget converts a ReasoningLevel to a concrete thinkingBudget
// value for Vertex AI using a dynamic percentage of min(maxOutputTokens, modelMax).
func mapReasoningToVertexBudget(level types.ReasoningLevel, maxOutputTokens int, modelID string) int {
	modelMax := maxThinkingTokensForVertexModel(modelID)
	cap := modelMax
	if maxOutputTokens > 0 && maxOutputTokens < cap {
		cap = maxOutputTokens
	}

	var pct float64
	switch level {
	case types.ReasoningMinimal:
		pct = 0.05
	case types.ReasoningLow:
		pct = 0.20
	case types.ReasoningMedium:
		pct = 0.40
	case types.ReasoningHigh:
		pct = 0.70
	case types.ReasoningXHigh:
		pct = 1.00
	default:
		pct = 0.40
	}

	budget := int(float64(cap) * pct)
	if budget < 1 {
		budget = 1
	}
	return budget
}

// isGemini3Model returns true for Gemini 3.x models (same logic as Google provider).
func isGemini3Model(modelID string) bool {
	lower := strings.ToLower(modelID)
	return strings.Contains(lower, "gemini-3.") ||
		strings.Contains(lower, "gemini-3-") ||
		lower == "gemini-3"
}

// isGemmaModel returns true for Gemma models.
// Gemma models do not support systemInstruction — matches TS isGemmaModel check.
func isGemmaModel(modelID string) bool {
	return strings.HasPrefix(strings.ToLower(modelID), "gemma-")
}

// mapReasoningToGemini3Level converts a ReasoningLevel to the thinkingLevel string
// used by Gemini 3 models. 'none' maps to 'minimal' (Gemini 3 cannot fully disable thinking).
func mapReasoningToGemini3Level(level types.ReasoningLevel) string {
	switch level {
	case types.ReasoningNone:
		return "minimal"
	case types.ReasoningMinimal:
		return "minimal"
	case types.ReasoningLow:
		return "low"
	case types.ReasoningMedium:
		return "medium"
	case types.ReasoningHigh, types.ReasoningXHigh:
		return "high"
	default:
		return "medium"
	}
}

// supportsFunctionResponseParts returns whether this model supports multimodal
// content in tool result function responses. Only Gemini 3+ models support this.
func (m *LanguageModel) supportsFunctionResponseParts() bool {
	return isGemini3Model(m.modelID)
}

// convertResponse converts a Vertex AI response to GenerateResult
// Uses the same response format as Google Generative AI
func (m *LanguageModel) convertResponse(response vertexResponse) *types.GenerateResult {
	result := &types.GenerateResult{
		Usage:       convertVertexUsage(response.UsageMetadata),
		RawResponse: response,
	}

	// Extract content from first candidate
	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]

		// Extract text from parts
		var textParts []string
		for _, part := range candidate.Content.Parts {
			// Thought parts with inlineData are reasoning files.
			if part.Thought && part.InlineData != nil {
				data, _ := base64.StdEncoding.DecodeString(part.InlineData.Data)
				result.Content = append(result.Content, types.ReasoningFileContent{
					MediaType: part.InlineData.MimeType,
					Data:      data,
				})
				continue
			}
			// Thought text parts → ReasoningContent with cryptographic signature.
			if part.Thought {
				if part.Text != "" {
					result.Content = append(result.Content, types.ReasoningContent{
						Text:      part.Text,
						Signature: part.ThoughtSignature,
					})
				}
				continue
			}
			if part.Text != "" {
				textParts = append(textParts, part.Text)
				// If this text part carries a thoughtSignature, attach it via
				// ProviderMetadata so callers can forward it in multi-turn history.
				// Use "vertex" key to match TS providerOptionsName for Vertex.
				if part.ThoughtSignature != "" {
					meta, _ := json.Marshal(map[string]interface{}{
						"vertex": map[string]interface{}{
							"thoughtSignature": part.ThoughtSignature,
						},
					})
					result.Content = append(result.Content, types.TextContent{
						Text:             part.Text,
						ProviderMetadata: meta,
					})
				}
			}
			// Handle function calls — capture ThoughtSignature for multi-turn forwarding.
			if part.FunctionCall != nil {
				result.ToolCalls = append(result.ToolCalls, types.ToolCall{
					ID:               part.FunctionCall.Name, // Vertex doesn't provide IDs
					ToolName:         part.FunctionCall.Name,
					Arguments:        part.FunctionCall.Args,
					ThoughtSignature: part.ThoughtSignature,
				})
			}
		}

		if len(textParts) > 0 {
			result.Text = textParts[0]
		}

		// Map finish reason — matches TS mapGoogleGenerativeAIFinishReason.
		hasToolCalls := len(result.ToolCalls) > 0
		switch candidate.FinishReason {
		case "STOP":
			if hasToolCalls {
				result.FinishReason = types.FinishReasonToolCalls
			} else {
				result.FinishReason = types.FinishReasonStop
			}
		case "MAX_TOKENS":
			result.FinishReason = types.FinishReasonLength
		case "IMAGE_SAFETY", "RECITATION", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
			result.FinishReason = types.FinishReasonContentFilter
		case "MALFORMED_FUNCTION_CALL":
			result.FinishReason = types.FinishReasonError
		default:
			result.FinishReason = types.FinishReasonOther
		}

		// Populate full ProviderMetadata under "vertex" key (matches TS providerOptionsName).
		{
			meta := map[string]json.RawMessage{}
			if response.PromptFeedback != nil {
				meta["promptFeedback"] = response.PromptFeedback
			}
			if candidate.GroundingMetadata != nil {
				meta["groundingMetadata"] = candidate.GroundingMetadata
			}
			if candidate.UrlContextMetadata != nil {
				meta["urlContextMetadata"] = candidate.UrlContextMetadata
			}
			if candidate.SafetyRatings != nil {
				meta["safetyRatings"] = candidate.SafetyRatings
			}
			if candidate.FinishMessage != "" {
				if fm, err := json.Marshal(candidate.FinishMessage); err == nil {
					meta["finishMessage"] = fm
				}
			}
			if response.UsageMetadata != nil {
				if um, err := json.Marshal(response.UsageMetadata); err == nil {
					meta["usageMetadata"] = um
				}
			}
			result.ProviderMetadata = map[string]interface{}{
				"vertex": meta,
			}
		}
	}

	return result
}

// handleError converts various errors to provider errors
func (m *LanguageModel) handleError(err error) error {
	return providererrors.NewProviderError("google-vertex", 0, "", err.Error(), err)
}

// convertVertexUsage converts Vertex AI usage to detailed Usage struct
func convertVertexUsage(usage *vertexUsageMetadata) types.Usage {
	if usage == nil {
		return types.Usage{}
	}

	promptTokens := int64(usage.PromptTokenCount)
	candidatesTokens := int64(usage.CandidatesTokenCount)
	cachedContentTokens := int64(usage.CachedContentTokenCount)
	thoughtsTokens := int64(usage.ThoughtsTokenCount)

	// Calculate totals
	totalOutputTokens := candidatesTokens + thoughtsTokens
	totalTokens := promptTokens + totalOutputTokens

	result := types.Usage{
		InputTokens:  &promptTokens,
		OutputTokens: &totalOutputTokens,
		TotalTokens:  &totalTokens,
	}

	// Parse text and image tokens from promptTokensDetails
	var textTokens *int64
	var imageTokens *int64
	if usage.PromptTokensDetails != nil && len(usage.PromptTokensDetails) > 0 {
		var textCount, imageCount int64
		for _, detail := range usage.PromptTokensDetails {
			switch detail.Modality {
			case "TEXT":
				textCount += int64(detail.TokenCount)
			case "IMAGE":
				imageCount += int64(detail.TokenCount)
			}
		}
		if textCount > 0 {
			textTokens = &textCount
		}
		if imageCount > 0 {
			imageTokens = &imageCount
		}
	}

	// Set input token details (cache information and text/image breakdown)
	if cachedContentTokens > 0 || textTokens != nil || imageTokens != nil {
		noCacheTokens := promptTokens - cachedContentTokens
		result.InputDetails = &types.InputTokenDetails{
			NoCacheTokens:    &noCacheTokens,
			CacheReadTokens:  &cachedContentTokens,
			CacheWriteTokens: nil, // Vertex doesn't report cache write tokens separately
			TextTokens:       textTokens,
			ImageTokens:      imageTokens,
		}
	}

	// Set output token details (text vs reasoning tokens)
	if thoughtsTokens > 0 {
		result.OutputDetails = &types.OutputTokenDetails{
			TextTokens:      &candidatesTokens,
			ReasoningTokens: &thoughtsTokens,
		}
	}

	// Store raw usage for provider-specific details
	result.Raw = map[string]interface{}{
		"promptTokenCount":     usage.PromptTokenCount,
		"candidatesTokenCount": usage.CandidatesTokenCount,
		"totalTokenCount":      usage.TotalTokenCount,
	}

	if usage.CachedContentTokenCount > 0 {
		result.Raw["cachedContentTokenCount"] = usage.CachedContentTokenCount
	}
	if usage.ThoughtsTokenCount > 0 {
		result.Raw["thoughtsTokenCount"] = usage.ThoughtsTokenCount
	}
	if usage.TrafficType != "" {
		result.Raw["trafficType"] = usage.TrafficType
	}

	return result
}

// vertexCandidate holds a single candidate from a Vertex AI response.
type vertexCandidate struct {
	Content struct {
		Parts []vertexPart `json:"parts"`
		Role  string       `json:"role"`
	} `json:"content"`
	FinishReason       string          `json:"finishReason"`
	FinishMessage      string          `json:"finishMessage,omitempty"`
	Index              int             `json:"index"`
	GroundingMetadata  json.RawMessage `json:"groundingMetadata,omitempty"`
	UrlContextMetadata json.RawMessage `json:"urlContextMetadata,omitempty"`
	SafetyRatings      json.RawMessage `json:"safetyRatings,omitempty"`
}

// vertexResponse represents the Vertex AI API response
// Uses the same format as Google Generative AI
type vertexResponse struct {
	Candidates     []vertexCandidate    `json:"candidates"`
	UsageMetadata  *vertexUsageMetadata `json:"usageMetadata,omitempty"`
	PromptFeedback json.RawMessage      `json:"promptFeedback,omitempty"`
}

// vertexUsageMetadata represents Vertex AI's usage information
type vertexUsageMetadata struct {
	PromptTokenCount        int    `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount    int    `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount         int    `json:"totalTokenCount,omitempty"`
	CachedContentTokenCount int    `json:"cachedContentTokenCount,omitempty"`
	ThoughtsTokenCount      int    `json:"thoughtsTokenCount,omitempty"`
	TrafficType             string `json:"trafficType,omitempty"`
	PromptTokensDetails     []struct {
		Modality   string `json:"modality,omitempty"`
		TokenCount int    `json:"tokenCount,omitempty"`
	} `json:"promptTokensDetails,omitempty"`
}

// vertexPart represents a part in Vertex AI's content structure
type vertexPart struct {
	Text             string `json:"text,omitempty"`
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	FunctionCall     *struct {
		Name string                 `json:"name"`
		Args map[string]interface{} `json:"args"`
	} `json:"functionCall,omitempty"`
	// InlineData carries base64-encoded binary content (e.g. reasoning files).
	InlineData *struct {
		MimeType string `json:"mimeType"`
		Data     string `json:"data"` // base64-encoded
	} `json:"inlineData,omitempty"`
}

// vertexStream implements provider.TextStream for Vertex AI streaming
type vertexStream struct {
	reader                 io.ReadCloser
	parser                 *streaming.SSEParser
	err                    error
	partBuffer             []vertexPart         // parts buffered from the current SSE event
	finishPending          *types.FinishReason  // finish reason to emit after partBuffer is drained
	lastFinishMessage      string               // finishMessage from the last finish candidate
	lastGroundingMetadata  json.RawMessage      // accumulated groundingMetadata
	lastUrlContextMetadata json.RawMessage      // accumulated urlContextMetadata
	lastSafetyRatings      json.RawMessage      // accumulated safetyRatings
	lastPromptFeedback     json.RawMessage      // promptFeedback from any chunk
	lastUsageMetadata      *vertexUsageMetadata // usageMetadata from the last chunk
}

// newVertexStream creates a new Vertex AI stream
func newVertexStream(reader io.ReadCloser) *vertexStream {
	return &vertexStream{
		reader: reader,
		parser: streaming.NewSSEParser(reader),
	}
}

// Close implements io.Closer
func (s *vertexStream) Close() error {
	return s.reader.Close()
}

// Next returns the next chunk in the stream
func (s *vertexStream) Next() (*provider.StreamChunk, error) {
	if s.err != nil {
		return nil, s.err
	}

	// Drain buffered parts from the current SSE event before reading a new one.
	if len(s.partBuffer) > 0 {
		part := s.partBuffer[0]
		s.partBuffer = s.partBuffer[1:]
		// Thought parts with inlineData are reasoning files.
		if part.Thought && part.InlineData != nil {
			data, _ := base64.StdEncoding.DecodeString(part.InlineData.Data)
			return &provider.StreamChunk{
				Type: provider.ChunkTypeReasoningFile,
				ReasoningFileContent: &types.ReasoningFileContent{
					MediaType: part.InlineData.MimeType,
					Data:      data,
				},
			}, nil
		}
		// Thought parts (reasoning) → ChunkTypeReasoning with optional ThoughtSignature.
		// Use "vertex" key to match TS providerOptionsName for Vertex.
		if part.Thought && part.Text != "" {
			chunk := &provider.StreamChunk{
				Type: provider.ChunkTypeReasoning,
				Text: part.Text,
			}
			if part.ThoughtSignature != "" {
				meta, _ := json.Marshal(map[string]interface{}{
					"vertex": map[string]interface{}{
						"thoughtSignature": part.ThoughtSignature,
					},
				})
				chunk.ProviderMetadata = meta
			}
			return chunk, nil
		}
		if part.Text != "" {
			return &provider.StreamChunk{
				Type: provider.ChunkTypeText,
				Text: part.Text,
			}, nil
		}
		// Function call parts → emit a ToolCall chunk with optional ThoughtSignature.
		if part.FunctionCall != nil {
			return &provider.StreamChunk{
				Type: provider.ChunkTypeToolCall,
				ToolCall: &types.ToolCall{
					ID:               part.FunctionCall.Name,
					ToolName:         part.FunctionCall.Name,
					Arguments:        part.FunctionCall.Args,
					ThoughtSignature: part.ThoughtSignature,
				},
			}, nil
		}
		// Unknown part type — skip it.
		return s.Next()
	}

	// Emit pending finish reason once all parts have been drained.
	if s.finishPending != nil {
		chunk := &provider.StreamChunk{
			Type:         provider.ChunkTypeFinish,
			FinishReason: *s.finishPending,
		}
		// Include full ProviderMetadata under "vertex" key (matches TS providerOptionsName).
		meta := map[string]json.RawMessage{}
		if s.lastPromptFeedback != nil {
			meta["promptFeedback"] = s.lastPromptFeedback
		}
		if s.lastGroundingMetadata != nil {
			meta["groundingMetadata"] = s.lastGroundingMetadata
		}
		if s.lastUrlContextMetadata != nil {
			meta["urlContextMetadata"] = s.lastUrlContextMetadata
		}
		if s.lastSafetyRatings != nil {
			meta["safetyRatings"] = s.lastSafetyRatings
		}
		if s.lastFinishMessage != "" {
			if fm, err := json.Marshal(s.lastFinishMessage); err == nil {
				meta["finishMessage"] = fm
			}
		}
		if s.lastUsageMetadata != nil {
			if um, err := json.Marshal(s.lastUsageMetadata); err == nil {
				meta["usageMetadata"] = um
			}
		}
		if len(meta) > 0 {
			provMeta, _ := json.Marshal(map[string]interface{}{"vertex": meta})
			chunk.ProviderMetadata = provMeta
		}
		s.finishPending = nil
		return chunk, nil
	}

	// Get next SSE event.
	event, err := s.parser.Next()
	if err != nil {
		s.err = err
		return nil, err
	}

	// Check for stream completion.
	if streaming.IsStreamDone(event) {
		s.err = io.EOF
		return nil, io.EOF
	}

	// Parse the event data as JSON.
	var chunkData vertexResponse
	if err := json.Unmarshal([]byte(event.Data), &chunkData); err != nil {
		return nil, fmt.Errorf("failed to parse stream chunk: %w", err)
	}

	// Accumulate top-level metadata.
	if chunkData.PromptFeedback != nil {
		s.lastPromptFeedback = chunkData.PromptFeedback
	}
	if chunkData.UsageMetadata != nil {
		s.lastUsageMetadata = chunkData.UsageMetadata
	}

	if len(chunkData.Candidates) > 0 {
		candidate := chunkData.Candidates[0]

		// Accumulate candidate-level metadata.
		if candidate.GroundingMetadata != nil {
			s.lastGroundingMetadata = candidate.GroundingMetadata
		}
		if candidate.UrlContextMetadata != nil {
			s.lastUrlContextMetadata = candidate.UrlContextMetadata
		}
		if candidate.SafetyRatings != nil {
			s.lastSafetyRatings = candidate.SafetyRatings
		}
		if candidate.FinishMessage != "" {
			s.lastFinishMessage = candidate.FinishMessage
		}

		// Save finish reason; emit it after all parts are drained.
		if candidate.FinishReason != "" {
			var fr types.FinishReason
			switch candidate.FinishReason {
			case "STOP":
				fr = types.FinishReasonStop
			case "MAX_TOKENS":
				fr = types.FinishReasonLength
			case "IMAGE_SAFETY", "RECITATION", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
				fr = types.FinishReasonContentFilter
			case "MALFORMED_FUNCTION_CALL":
				fr = types.FinishReasonError
			default:
				fr = types.FinishReasonOther
			}
			s.finishPending = &fr
		}

		// Buffer all parts from this event so they are emitted one at a time.
		if len(candidate.Content.Parts) > 0 {
			s.partBuffer = candidate.Content.Parts
			return s.Next()
		}

		// No parts — emit finish reason if pending, otherwise get next event.
		if s.finishPending != nil {
			return s.Next()
		}
	}

	// Empty event — get next.
	return s.Next()
}

// Err returns any error that occurred during streaming
func (s *vertexStream) Err() error {
	if s.err == io.EOF {
		return nil
	}
	return s.err
}
