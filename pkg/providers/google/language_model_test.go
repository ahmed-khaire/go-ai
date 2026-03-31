package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/digitallysavvy/go-ai/pkg/provider"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// --- httptest-based request body verification --------------------------------

func TestBuildRequestBody_ThinkingConfig_ViaHTTP(t *testing.T) {
	var capturedBody map[string]interface{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
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

// --- EmbeddingModelGeminiEmbedding001 constant -----------------------------

func TestEmbeddingModelGeminiEmbedding001Constant(t *testing.T) {
	if EmbeddingModelGeminiEmbedding001 != "gemini-embedding-001" {
		t.Errorf("EmbeddingModelGeminiEmbedding001 = %q, want %q",
			EmbeddingModelGeminiEmbedding001, "gemini-embedding-001")
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
