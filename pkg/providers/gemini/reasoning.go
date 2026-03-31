package gemini

import (
	"strings"

	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// isGemini3Model reports whether modelID identifies a Gemini 3.x model.
// Matches the TS SDK pattern: /gemini-3[\.\-]/i or /gemini-3$/i.
func isGemini3Model(modelID string) bool {
	lower := strings.ToLower(modelID)
	return strings.Contains(lower, "gemini-3.") ||
		strings.Contains(lower, "gemini-3-") ||
		lower == "gemini-3"
}

// isGemmaModel reports whether modelID identifies a Gemma model.
// Gemma models do not support systemInstruction.
func isGemmaModel(modelID string) bool {
	return strings.HasPrefix(strings.ToLower(modelID), "gemma-")
}

// mapReasoningToGemini3Level converts a ReasoningLevel to the thinkingLevel
// string accepted by Gemini 3 models. ReasoningNone maps to "minimal" because
// Gemini 3 cannot fully disable thinking.
func mapReasoningToGemini3Level(level types.ReasoningLevel) string {
	switch level {
	case types.ReasoningNone, types.ReasoningMinimal:
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

// maxThinkingTokensForModel returns the maximum thinking token budget for a
// Gemini model. Mirrors getMaxThinkingTokensForGemini25Model in the TS SDK:
//   - gemini-2.5-pro and gemini-3-pro-image variants → 32768
//   - gemini-2.5-flash → 24576
//   - gemini-2.0-flash-thinking → 8192 (model-specific cap)
//   - all other models → 8192 safe default
func maxThinkingTokensForModel(modelID string) int {
	lower := strings.ToLower(modelID)
	switch {
	case strings.Contains(lower, "gemini-2.5-pro"):
		return 32768
	case strings.Contains(lower, "gemini-3-pro-image"):
		return 32768
	case strings.Contains(lower, "gemini-2.5-flash"):
		return 24576
	case strings.Contains(lower, "gemini-2.0-flash-thinking"):
		return 8192
	default:
		return 8192
	}
}

// mapReasoningBudget converts a ReasoningLevel to a thinkingBudget integer for
// Gemini 2.x models. The budget is a percentage of min(maxOutputTokens, modelMax).
// Percentages: minimal=2%, low=10%, medium=30%, high=60%, xhigh=90%.
// The result is floored at 1024 tokens.
func mapReasoningBudget(level types.ReasoningLevel, maxOutputTokens int, modelID string) int {
	modelMax := maxThinkingTokensForModel(modelID)
	cap := modelMax
	if maxOutputTokens > 0 && maxOutputTokens < cap {
		cap = maxOutputTokens
	}

	var pct float64
	switch level {
	case types.ReasoningMinimal:
		pct = 0.02
	case types.ReasoningLow:
		pct = 0.10
	case types.ReasoningMedium:
		pct = 0.30
	case types.ReasoningHigh:
		pct = 0.60
	case types.ReasoningXHigh:
		pct = 0.90
	default:
		pct = 0.30
	}

	budget := int(float64(cap) * pct)
	if budget < 1024 {
		budget = 1024
	}
	if budget > modelMax {
		budget = modelMax
	}
	return budget
}
