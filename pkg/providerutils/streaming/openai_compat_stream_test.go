package streaming

import (
	"io"
	"strings"
	"testing"

	"github.com/digitallysavvy/go-ai/pkg/provider"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// mapTestFinishReason is the standard mapper used in these tests.
func mapTestFinishReason(reason string) types.FinishReason {
	switch reason {
	case "stop":
		return types.FinishReasonStop
	case "tool_calls":
		return types.FinishReasonToolCalls
	case "length":
		return types.FinishReasonLength
	default:
		return types.FinishReasonOther
	}
}

func newTestStream(sseData string) *OpenAICompatStream {
	return NewOpenAICompatStream(io.NopCloser(strings.NewReader(sseData)), mapTestFinishReason)
}

func TestOpenAICompatStream_TextChunks(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	stream := newTestStream(sseData)
	defer stream.Close() //nolint:errcheck

	var chunks []*provider.StreamChunk
	for {
		chunk, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkTypeText || chunks[0].Text != "Hello" {
		t.Errorf("chunk[0]: got type=%v text=%q", chunks[0].Type, chunks[0].Text)
	}
	if chunks[1].Type != provider.ChunkTypeText || chunks[1].Text != " world" {
		t.Errorf("chunk[1]: got type=%v text=%q", chunks[1].Type, chunks[1].Text)
	}
	if chunks[2].Type != provider.ChunkTypeFinish || chunks[2].FinishReason != types.FinishReasonStop {
		t.Errorf("chunk[2]: expected finish/stop, got type=%v reason=%v", chunks[2].Type, chunks[2].FinishReason)
	}
}

// TestOpenAICompatStream_ToolCallAccumulation verifies that tool call arguments
// are accumulated across deltas and emitted only at finish_reason, not per-delta.
func TestOpenAICompatStream_ToolCallAccumulation(t *testing.T) {
	// Three deltas: first carries id+name+empty args, second carries partial args
	// {"ready":true} (valid JSON mid-stream), third carries finish_reason.
	// The fix requires that no tool call is emitted until the finish_reason delta.
	sseData := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"fn","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ready\":true}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	stream := newTestStream(sseData)
	defer stream.Close() //nolint:errcheck

	var chunks []*provider.StreamChunk
	for {
		chunk, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks (tool_call + finish), got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkTypeToolCall {
		t.Fatalf("chunk[0]: expected tool_call, got %v", chunks[0].Type)
	}
	if chunks[0].ToolCall.ID != "call_1" {
		t.Errorf("tool call id: got %q", chunks[0].ToolCall.ID)
	}
	if chunks[0].ToolCall.ToolName != "fn" {
		t.Errorf("tool call name: got %q", chunks[0].ToolCall.ToolName)
	}
	if chunks[0].ToolCall.Arguments["ready"] != true {
		t.Errorf("tool call arg ready: got %v", chunks[0].ToolCall.Arguments["ready"])
	}
	if chunks[1].Type != provider.ChunkTypeFinish {
		t.Errorf("chunk[1]: expected finish, got %v", chunks[1].Type)
	}
	if chunks[1].FinishReason != types.FinishReasonToolCalls {
		t.Errorf("finish reason: got %v", chunks[1].FinishReason)
	}
}

// TestOpenAICompatStream_ToolCallPartialJSON verifies that even partial JSON
// arguments that would fail to unmarshal are handled gracefully (nil args).
func TestOpenAICompatStream_ToolCallPartialJSON(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"calc","arguments":"{\"op\":"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"add\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	stream := newTestStream(sseData)
	defer stream.Close() //nolint:errcheck

	var chunks []*provider.StreamChunk
	for {
		chunk, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkTypeToolCall {
		t.Fatalf("expected tool_call chunk, got %v", chunks[0].Type)
	}
	if chunks[0].ToolCall.Arguments["op"] != "add" {
		t.Errorf("expected op=add, got %v", chunks[0].ToolCall.Arguments["op"])
	}
	if chunks[1].Type != provider.ChunkTypeFinish {
		t.Errorf("expected finish chunk, got %v", chunks[1].Type)
	}
}

// TestOpenAICompatStream_MultipleToolCalls verifies that multiple tool calls
// (different indices) are all flushed correctly at finish_reason.
func TestOpenAICompatStream_MultipleToolCalls(t *testing.T) {
	sseData := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"f0","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c1","type":"function","function":{"name":"f1","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"b\":2}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	stream := newTestStream(sseData)
	defer stream.Close() //nolint:errcheck

	var chunks []*provider.StreamChunk
	for {
		chunk, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks = append(chunks, chunk)
	}

	// Expect: tool_call[0], tool_call[1], finish
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}
	if chunks[0].Type != provider.ChunkTypeToolCall || chunks[0].ToolCall.ID != "c0" {
		t.Errorf("chunk[0]: expected tool_call c0, got %v/%v", chunks[0].Type, chunks[0].ToolCall)
	}
	if chunks[1].Type != provider.ChunkTypeToolCall || chunks[1].ToolCall.ID != "c1" {
		t.Errorf("chunk[1]: expected tool_call c1, got %v/%v", chunks[1].Type, chunks[1].ToolCall)
	}
	if chunks[2].Type != provider.ChunkTypeFinish {
		t.Errorf("chunk[2]: expected finish, got %v", chunks[2].Type)
	}
}
