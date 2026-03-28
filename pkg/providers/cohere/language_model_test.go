package cohere

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/digitallysavvy/go-ai/pkg/provider"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// v2 mock response JSON
const cohereV2MockResponse = `{"generation_id":"test","message":{"role":"assistant","content":[{"type":"text","text":"hello"}],"tool_calls":null},"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":5,"output_tokens":3}}}`

func TestCohereNoWarningWhenReasoningNil(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(cohereV2MockResponse))
	}))
	defer srv.Close()

	prov := New(Config{BaseURL: srv.URL, APIKey: "test-key"})
	model := NewLanguageModel(prov, "command-r-plus")

	opts := &provider.GenerateOptions{}
	result, err := model.DoGenerate(t.Context(), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Warnings) != 0 {
		t.Errorf("expected no warnings when Reasoning is nil, got: %+v", result.Warnings)
	}
	if result.Text != "hello" {
		t.Errorf("expected text 'hello', got: %q", result.Text)
	}
}

func TestCohereReasoningNoneDisablesThinking(t *testing.T) {
	model := &LanguageModel{modelID: "command-r-plus"}
	level := types.ReasoningNone
	opts := &provider.GenerateOptions{Reasoning: &level}
	body := model.buildRequestBody(opts)
	thinking, ok := body["thinking"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected thinking field to be a map, got: %T", body["thinking"])
	}
	if thinking["type"] != "disabled" {
		t.Errorf("expected thinking.type='disabled', got: %v", thinking["type"])
	}
}

func TestCohereReasoningHighBudget(t *testing.T) {
	model := &LanguageModel{modelID: "command-r-plus"}
	level := types.ReasoningHigh
	opts := &provider.GenerateOptions{Reasoning: &level}
	body := model.buildRequestBody(opts)
	thinking, ok := body["thinking"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected thinking field to be a map, got: %T", body["thinking"])
	}
	if thinking["type"] != "enabled" {
		t.Errorf("expected thinking.type='enabled', got: %v", thinking["type"])
	}
	if thinking["token_budget"] != 19661 {
		t.Errorf("expected token_budget=19661, got: %v", thinking["token_budget"])
	}
}

func TestCohereReasoningDefaultOmitted(t *testing.T) {
	model := &LanguageModel{modelID: "command-r-plus"}
	level := types.ReasoningDefault
	opts := &provider.GenerateOptions{Reasoning: &level}
	body := model.buildRequestBody(opts)
	if _, ok := body["thinking"]; ok {
		t.Errorf("expected no thinking field when Reasoning is ReasoningDefault, got: %v", body["thinking"])
	}
}
