package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	internalhttp "github.com/digitallysavvy/go-ai/pkg/internal/http"
	providererrors "github.com/digitallysavvy/go-ai/pkg/provider/errors"
	"github.com/digitallysavvy/go-ai/pkg/provider"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
	"github.com/digitallysavvy/go-ai/pkg/providerutils/prompt"
	"github.com/digitallysavvy/go-ai/pkg/providerutils/streaming"
	"github.com/digitallysavvy/go-ai/pkg/providerutils/tool"
)

// LanguageModel implements the provider.LanguageModel interface for Google (Gemini)
type LanguageModel struct {
	provider *Provider
	modelID  string
}

// NewLanguageModel creates a new Google language model
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
	return "google"
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
	// Gemini Pro Vision supports images
	return m.modelID == "gemini-pro-vision" ||
		m.modelID == "gemini-1.5-pro" ||
		m.modelID == "gemini-1.5-flash"
}

// supportsFunctionResponseParts returns whether this model supports
// multimodal content in tool result function responses.
// Only Gemini 3+ models support image/file parts in functionResponse.parts[].
func (m *LanguageModel) supportsFunctionResponseParts() bool {
	return strings.HasPrefix(m.modelID, "gemini-3")
}

// DoGenerate performs non-streaming text generation
func (m *LanguageModel) DoGenerate(ctx context.Context, opts *provider.GenerateOptions) (*types.GenerateResult, error) {
	// Build request body
	reqBody := m.buildRequestBody(opts)

	// Build path with API key
	path := fmt.Sprintf("/v1beta/models/%s:generateContent?key=%s", m.modelID, m.provider.APIKey())

	// Make API request
	var response googleResponse
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

	// Build path with API key
	path := fmt.Sprintf("/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", m.modelID, m.provider.APIKey())

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
	return newGoogleStream(httpResp.Body), nil
}

// buildRequestBody builds the Google API request body
func (m *LanguageModel) buildRequestBody(opts *provider.GenerateOptions) map[string]interface{} {
	body := map[string]interface{}{}

	// Convert messages to Google format
	if opts.Prompt.IsMessages() {
		body["contents"] = prompt.ToGoogleMessages(opts.Prompt.Messages, m.supportsFunctionResponseParts())
	} else if opts.Prompt.IsSimple() {
		body["contents"] = prompt.ToGoogleMessages(prompt.SimpleTextToMessages(opts.Prompt.Text), false)
	}

	// Add system instruction if present.
	// Gemma models do not support systemInstruction — matches TS isGemmaModel check (line 213).
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

	// Map top-level Reasoning to Google thinkingConfig.
	// Gemini 3 models use thinkingLevel ('minimal'/'low'/'medium'/'high').
	// Gemini 2.x and earlier use thinkingBudget (integer token count).
	// Call-level Reasoning takes precedence over ProviderOptions["google"].thinkingConfig.
	// provider-default → omit (use model default).
	if opts.Reasoning != nil && *opts.Reasoning != types.ReasoningDefault {
		// Gemini 3 image models (gemini-3-pro-image-*, gemini-3.1-flash-image-*)
		// do not support extended thinking — treat them like Gemini 2.x.
		// Matches TS: isGemini3Model(modelId) && !modelId.includes('gemini-3-pro-image')
		if isGemini3Model(m.modelID) && !strings.Contains(m.modelID, "image") {
			// Gemini 3 (non-image): use thinkingLevel string. 'none' maps to 'minimal'.
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
				budget := mapReasoningToGoogleBudget(*opts.Reasoning, maxOut, m.modelID)
				genConfig["thinkingConfig"] = map[string]interface{}{"thinkingBudget": budget}
			}
		}
	} else {
		// Fall back to ProviderOptions["google"].thinkingConfig when Reasoning is unset.
		// Matches TS: generationConfig.thinkingConfig = googleOptions?.thinkingConfig
		if opts.ProviderOptions != nil {
			if googleOpts, ok := opts.ProviderOptions["google"].(map[string]interface{}); ok {
				if thinkingConfig, ok := googleOpts["thinkingConfig"].(map[string]interface{}); ok {
					genConfig["thinkingConfig"] = thinkingConfig
				}
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
			structuredOutputs := true
			if opts.ProviderOptions != nil {
				if googleOpts, ok := opts.ProviderOptions["google"].(map[string]interface{}); ok {
					if so, ok := googleOpts["structuredOutputs"].(bool); ok {
						structuredOutputs = so
					}
				}
			}
			if structuredOutputs {
				genConfig["responseSchema"] = opts.ResponseFormat.Schema
			}
		}
	}

	// Forward additional provider options from ProviderOptions["google"] into
	// generationConfig and the top-level request body. Matches TS getArgs().
	var googleOpts map[string]interface{}
	if opts.ProviderOptions != nil {
		googleOpts, _ = opts.ProviderOptions["google"].(map[string]interface{})
	}
	if googleOpts != nil {
		// generationConfig fields
		for _, key := range []string{"responseModalities", "mediaResolution", "audioTimestamp", "imageConfig"} {
			if v, ok := googleOpts[key]; ok {
				genConfig[key] = v
			}
		}
		// Top-level request body fields
		if v, ok := googleOpts["safetySettings"]; ok {
			body["safetySettings"] = v
		}
		if v, ok := googleOpts["cachedContent"]; ok {
			body["cachedContent"] = v
		}
		if v, ok := googleOpts["labels"]; ok {
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
				if entry := buildGoogleNativeToolEntry(t); entry != nil {
					nativeEntries = append(nativeEntries, entry)
				}
			} else {
				functionTools = append(functionTools, t)
			}
		}

		if len(nativeEntries) > 0 {
			// Native provider tools are placed directly in the tools array.
			// retrievalConfig from provider options is merged into toolConfig when
			// using native tools (e.g. for location-based grounding).
			body["tools"] = nativeEntries
			if googleOpts != nil {
				if rc, ok := googleOpts["retrievalConfig"]; ok {
					toolConfig := map[string]interface{}{"retrievalConfig": rc}
					body["toolConfig"] = toolConfig
				}
			}
		} else if len(functionTools) > 0 {
			body["tools"] = []map[string]interface{}{
				{"functionDeclarations": tool.ToGoogleFormat(functionTools)},
			}

			// Determine functionCallingConfig mode based on ToolChoice and strict tools.
			// Matches TS prepareTools logic exactly:
			//   toolChoice=none → NONE; required/tool → ANY (VALIDATED if strict);
			//   auto or unset → AUTO (VALIDATED if strict).
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

// convertResponse converts a Google response to GenerateResult
// Updated in v6.0 to support detailed usage tracking
func (m *LanguageModel) convertResponse(response googleResponse) *types.GenerateResult {
	result := &types.GenerateResult{
		Usage:       convertGoogleUsage(response.UsageMetadata),
		RawResponse: response,
	}

	// Extract content from first candidate
	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]

		// Extract text from parts
		var textParts []string
		// lastCodeExecID links an executableCode part to its codeExecutionResult.
		var lastCodeExecID string
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
			// Non-thought inlineData parts → GeneratedFileContent.
			// Matches TS: 'inlineData' in part && !part.thought → type:'file'.
			if part.InlineData != nil {
				data, _ := base64.StdEncoding.DecodeString(part.InlineData.Data)
				result.Content = append(result.Content, types.GeneratedFileContent{
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
			// Code execution parts: emit as provider-executed tool call + tool result.
			// Matches TS doGenerate: executableCode → tool-call (providerExecuted),
			// codeExecutionResult → tool-result content block.
			if part.ExecutableCode != nil && part.ExecutableCode.Code != "" {
				toolCallID := fmt.Sprintf("code-exec-%d", len(result.ToolCalls)+1)
				lastCodeExecID = toolCallID
				result.ToolCalls = append(result.ToolCalls, types.ToolCall{
					ID:               toolCallID,
					ToolName:         "code_execution",
					Arguments:        map[string]interface{}{"code": part.ExecutableCode.Code, "language": part.ExecutableCode.Language},
					ProviderExecuted: true,
				})
				continue
			}
			if part.CodeExecutionResult != nil && lastCodeExecID != "" {
				result.Content = append(result.Content, types.ToolResultContent{
					ToolCallID: lastCodeExecID,
					ToolName:   "code_execution",
					Result: map[string]interface{}{
						"outcome": part.CodeExecutionResult.Outcome,
						"output":  part.CodeExecutionResult.Output,
					},
				})
				lastCodeExecID = ""
				continue
			}
			// Text parts → always added to Content. ThoughtSignature forwarded via ProviderMetadata.
			// Matches TS: content.push({type:'text', text, providerMetadata}).
			if part.Text != "" {
				textParts = append(textParts, part.Text)
				tc := types.TextContent{Text: part.Text}
				if part.ThoughtSignature != "" {
					meta, _ := json.Marshal(map[string]interface{}{
						"google": map[string]interface{}{
							"thoughtSignature": part.ThoughtSignature,
						},
					})
					tc.ProviderMetadata = meta
				}
				result.Content = append(result.Content, tc)
			}
			// Handle function calls — capture ThoughtSignature so multi-turn callers
			// can forward the sealed reasoning chain alongside the function call.
			if part.FunctionCall != nil {
				result.ToolCalls = append(result.ToolCalls, types.ToolCall{
					ID:               part.FunctionCall.Name, // Google doesn't provide IDs
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
		// STOP with tool calls → tool-calls; content safety variants → content-filter;
		// MALFORMED_FUNCTION_CALL → error.
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

		// Populate full ProviderMetadata matching TS doGenerate:
		// providerMetadata.google.{promptFeedback, groundingMetadata, urlContextMetadata,
		//                          safetyRatings, usageMetadata, finishMessage}
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
				"google": meta,
			}
		}
	}

	return result
}

// handleError converts various errors to provider errors
func (m *LanguageModel) handleError(err error) error {
	return providererrors.NewProviderError("google", 0, "", err.Error(), err)
}

// maxThinkingTokensForModel returns the maximum thinking token capacity for a Google model.
// These values mirror the Google AI SDK model capability table.
func maxThinkingTokensForModel(modelID string) int {
	switch {
	case strings.Contains(modelID, "gemini-2.5-pro"):
		return 32768
	case strings.Contains(modelID, "gemini-2.5-flash"):
		return 24576
	case strings.Contains(modelID, "gemini-2.0-flash-thinking"):
		return 8192
	default:
		return 8192 // conservative fallback for unknown models
	}
}

// mapReasoningToGoogleBudget converts a ReasoningLevel to a concrete thinkingBudget
// value using a dynamic percentage of min(maxOutputTokens, modelMaxThinking).
// Percentages: minimal=5%, low=20%, medium=40%, high=70%, xhigh=100%.
// Minimum 1 token for non-zero levels.
func mapReasoningToGoogleBudget(level types.ReasoningLevel, maxOutputTokens int, modelID string) int {
	modelMax := maxThinkingTokensForModel(modelID)
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

// isGemini3Model returns true for Gemini 3.x models.
// Matches the TS SDK: /gemini-3[\.\-]/i.test(modelId) || /gemini-3$/i.test(modelId)
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

// convertGoogleUsage converts Google usage to detailed Usage struct
// Implements v6.0 detailed token tracking with cache and reasoning tokens
func convertGoogleUsage(usage *googleUsageMetadata) types.Usage {
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
	// Google provides cachedContentTokenCount for cache reads
	if cachedContentTokens > 0 || textTokens != nil || imageTokens != nil {
		noCacheTokens := promptTokens - cachedContentTokens
		result.InputDetails = &types.InputTokenDetails{
			NoCacheTokens:   &noCacheTokens,
			CacheReadTokens: &cachedContentTokens,
			// Google doesn't report cache write tokens separately
			CacheWriteTokens: nil,
			TextTokens:       textTokens,
			ImageTokens:      imageTokens,
		}
	}

	// Set output token details (text vs reasoning tokens)
	// Google provides thoughtsTokenCount for reasoning in Gemini thinking models
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

// googleCandidate holds a single candidate from a Google API response.
type googleCandidate struct {
	Content struct {
		Parts []googlePart `json:"parts"`
		Role  string       `json:"role"`
	} `json:"content"`
	FinishReason       string          `json:"finishReason"`
	FinishMessage      string          `json:"finishMessage,omitempty"`
	Index              int             `json:"index"`
	GroundingMetadata  json.RawMessage `json:"groundingMetadata,omitempty"`
	UrlContextMetadata json.RawMessage `json:"urlContextMetadata,omitempty"`
	SafetyRatings      json.RawMessage `json:"safetyRatings,omitempty"`
}

// googleResponse represents the Google API response
// Updated in v6.0 to support detailed usage tracking
type googleResponse struct {
	Candidates     []googleCandidate    `json:"candidates"`
	UsageMetadata  *googleUsageMetadata `json:"usageMetadata,omitempty"`
	PromptFeedback json.RawMessage      `json:"promptFeedback,omitempty"`
}

// googleUsageMetadata represents Google's usage information with detailed token tracking
type googleUsageMetadata struct {
	PromptTokenCount        int    `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount    int    `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount         int    `json:"totalTokenCount,omitempty"`
	CachedContentTokenCount int    `json:"cachedContentTokenCount,omitempty"` // v6.0 - cache read
	ThoughtsTokenCount      int    `json:"thoughtsTokenCount,omitempty"`      // v6.0 - reasoning tokens
	TrafficType             string `json:"trafficType,omitempty"`             // v6.0 - metadata
	PromptTokensDetails     []struct {
		Modality   string `json:"modality,omitempty"`
		TokenCount int    `json:"tokenCount,omitempty"`
	} `json:"promptTokensDetails,omitempty"` // v6.0 - text/image token breakdown
}

// googlePart represents a part in Google's content structure
type googlePart struct {
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
	// ExecutableCode is set when the model requests code execution.
	ExecutableCode *struct {
		Language string `json:"language"`
		Code     string `json:"code"`
	} `json:"executableCode,omitempty"`
	// CodeExecutionResult is set when the model returns a code execution result.
	CodeExecutionResult *struct {
		Outcome string `json:"outcome"`
		Output  string `json:"output"`
	} `json:"codeExecutionResult,omitempty"`
}

// googleStream implements provider.TextStream for Google streaming.
// Emits structured block-boundary chunks (text-start/delta/end, reasoning-start/delta/end)
// and tool-input sequences (tool-input-start/delta/end + tool-call) to match the TS SDK.
type googleStream struct {
	reader  io.ReadCloser
	parser  *streaming.SSEParser
	err     error
	// Pre-converted chunks ready to emit; Next() drains this before reading the next SSE event.
	chunkBuffer []*provider.StreamChunk
	// Block boundary state tracked across SSE events (mirrors TS currentTextBlockId / currentReasoningBlockId).
	currentTextBlockID      string
	currentReasoningBlockID string
	blockCounter            int
	// Whether any non-provider-executed tool calls have been seen (affects STOP → tool-calls mapping).
	hasToolCalls bool
	// Code execution linking (Google-specific).
	codeExecCount  int
	lastCodeExecID string
	// Metadata accumulation across SSE events.
	lastGroundingMetadata  json.RawMessage
	lastUrlContextMetadata json.RawMessage
	lastSafetyRatings      json.RawMessage
	lastFinishMessage      string
	lastPromptFeedback     json.RawMessage
	lastUsageMetadata      *googleUsageMetadata
}

// newGoogleStream creates a new Google stream
func newGoogleStream(reader io.ReadCloser) *googleStream {
	return &googleStream{
		reader: reader,
		parser: streaming.NewSSEParser(reader),
	}
}

// Read implements io.Reader
func (s *googleStream) Read(p []byte) (n int, err error) {
	return s.reader.Read(p)
}

// Close implements io.Closer
func (s *googleStream) Close() error {
	return s.reader.Close()
}

// Next returns the next chunk in the stream
func (s *googleStream) Next() (*provider.StreamChunk, error) {
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
		// Code execution parts: emit as ToolCall (executableCode) or ToolResult (codeExecutionResult).
		// Matches TS doStream: executableCode → tool-call with providerExecuted, result → tool-result.
		if part.ExecutableCode != nil && part.ExecutableCode.Code != "" {
			s.codeExecCount++
			toolCallID := fmt.Sprintf("code-exec-%d", s.codeExecCount)
			s.lastCodeExecID = toolCallID
			return &provider.StreamChunk{
				Type: provider.ChunkTypeToolCall,
				ToolCall: &types.ToolCall{
					ID:               toolCallID,
					ToolName:         "code_execution",
					Arguments:        map[string]interface{}{"code": part.ExecutableCode.Code, "language": part.ExecutableCode.Language},
					ProviderExecuted: true,
				},
			}, nil
		}
		if part.CodeExecutionResult != nil && s.lastCodeExecID != "" {
			toolCallID := s.lastCodeExecID
			s.lastCodeExecID = ""
			return &provider.StreamChunk{
				Type: provider.ChunkTypeToolResult,
				ToolResult: &types.ToolResult{
					ToolCallID: toolCallID,
					ToolName:   "code_execution",
					Result: map[string]interface{}{
						"outcome": part.CodeExecutionResult.Outcome,
						"output":  part.CodeExecutionResult.Output,
					},
				},
			}, nil
		}
		// Thought parts (reasoning) map to ChunkTypeReasoning.
		// Carry the thoughtSignature in ProviderMetadata when present.
		if part.Thought && part.Text != "" {
			chunk := &provider.StreamChunk{
				Type: provider.ChunkTypeReasoning,
				Text: part.Text,
			}
			if part.ThoughtSignature != "" {
				meta, _ := json.Marshal(map[string]interface{}{
					"google": map[string]interface{}{
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
		// Include full ProviderMetadata matching TS doStream flush handler:
		// {promptFeedback, groundingMetadata, urlContextMetadata, safetyRatings,
		//  usageMetadata, finishMessage}
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
			provMeta, _ := json.Marshal(map[string]interface{}{"google": meta})
			chunk.ProviderMetadata = provMeta
		}
		s.finishPending = nil
		return chunk, nil
	}

	// Read next SSE event.
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
	var chunkData googleResponse
	if err := json.Unmarshal([]byte(event.Data), &chunkData); err != nil {
		return nil, fmt.Errorf("failed to parse stream chunk: %w", err)
	}

	// Accumulate top-level metadata (promptFeedback, usageMetadata).
	if chunkData.PromptFeedback != nil {
		s.lastPromptFeedback = chunkData.PromptFeedback
	}
	if chunkData.UsageMetadata != nil {
		s.lastUsageMetadata = chunkData.UsageMetadata
	}

	if len(chunkData.Candidates) > 0 {
		candidate := chunkData.Candidates[0]

		// Accumulate grounding/safety metadata across all stream chunks.
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
		// Matches TS mapGoogleGenerativeAIFinishReason (hasToolCalls handled at flush).
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
func (s *googleStream) Err() error {
	if s.err == io.EOF {
		return nil
	}
	return s.err
}
