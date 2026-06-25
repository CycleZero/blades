package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/CycleZero/blades"
	openaisdk "github.com/openai/openai-go/v3"
)

func TestToChatCompletionParamsAssistantRole(t *testing.T) {
	t.Parallel()

	model := &chatModel{model: "gpt-test"}
	req := &blades.ModelRequest{
		Messages: []*blades.Message{
			blades.UserMessage("hello"),
			blades.AssistantMessage("world"),
		},
	}
	params, err := model.toChatCompletionParams(false, req)
	if err != nil {
		t.Fatalf("toChatCompletionParams returned error: %v", err)
	}

	payload, err := json.Marshal(params.Messages)
	if err != nil {
		t.Fatalf("marshal params messages: %v", err)
	}
	if got, want := bytes.Count(payload, []byte(`"role":"assistant"`)), 1; got != want {
		t.Fatalf("assistant role count = %d, want %d; payload=%s", got, want, payload)
	}
	if got, want := bytes.Count(payload, []byte(`"role":"user"`)), 1; got != want {
		t.Fatalf("user role count = %d, want %d; payload=%s", got, want, payload)
	}
}

func TestChoiceToResponseMarksToolPartsIncomplete(t *testing.T) {
	t.Parallel()

	response, err := choiceToResponse(context.Background(), openaisdk.ChatCompletionNewParams{}, &openaisdk.ChatCompletion{
		Choices: []openaisdk.ChatCompletionChoice{
			{
				Message: openaisdk.ChatCompletionMessage{
					ToolCalls: []openaisdk.ChatCompletionMessageToolCallUnion{
						{
							ID:   "call_1",
							Type: "function",
							Function: openaisdk.ChatCompletionMessageFunctionToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"city":"Paris"}`,
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("choiceToResponse returned error: %v", err)
	}

	toolPart, ok := response.Message.Parts[0].(blades.ToolPart)
	if !ok {
		t.Fatalf("part type = %T, want blades.ToolPart", response.Message.Parts[0])
	}
	if got, want := toolPart.Completed, false; got != want {
		t.Fatalf("tool completed = %t, want %t", got, want)
	}
}

func TestChunkChoiceToResponseMarksToolPartsIncomplete(t *testing.T) {
	t.Parallel()

	response, err := chunkChoiceToResponse(context.Background(), []openaisdk.ChatCompletionChunkChoice{
		{
			Delta: openaisdk.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openaisdk.ChatCompletionChunkChoiceDeltaToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: openaisdk.ChatCompletionChunkChoiceDeltaToolCallFunction{
							Name:      "get_weather",
							Arguments: `{"city":"Paris"}`,
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("chunkChoiceToResponse returned error: %v", err)
	}

	toolPart, ok := response.Message.Parts[0].(blades.ToolPart)
	if !ok {
		t.Fatalf("part type = %T, want blades.ToolPart", response.Message.Parts[0])
	}
	if got, want := toolPart.Completed, false; got != want {
		t.Fatalf("tool completed = %t, want %t", got, want)
	}
}

func TestChoiceToResponseExtractsReasoning(t *testing.T) {
	t.Parallel()

	raw := `{
		"id":"c1","object":"chat.completion","created":0,"model":"deepseek-reasoner",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"42","reasoning_content":"let me think"}}],
		"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}
	}`
	var cc openaisdk.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &cc); err != nil {
		t.Fatalf("unmarshal chat completion: %v", err)
	}

	resp, err := choiceToResponse(context.Background(), openaisdk.ChatCompletionNewParams{}, &cc)
	if err != nil {
		t.Fatalf("choiceToResponse returned error: %v", err)
	}
	if got, want := resp.Message.Reasoning(), "let me think"; got != want {
		t.Fatalf("Reasoning() = %q, want %q", got, want)
	}
	if got, want := resp.Message.Text(), "42"; got != want {
		t.Fatalf("Text() = %q, want %q", got, want)
	}
}

func TestChunkChoiceToResponseExtractsReasoning(t *testing.T) {
	t.Parallel()

	raw := `{"index":0,"delta":{"role":"assistant","reasoning_content":"hmm"}}`
	var choice openaisdk.ChatCompletionChunkChoice
	if err := json.Unmarshal([]byte(raw), &choice); err != nil {
		t.Fatalf("unmarshal chunk choice: %v", err)
	}

	resp, err := chunkChoiceToResponse(context.Background(), []openaisdk.ChatCompletionChunkChoice{choice})
	if err != nil {
		t.Fatalf("chunkChoiceToResponse returned error: %v", err)
	}
	if got, want := resp.Message.Reasoning(), "hmm"; got != want {
		t.Fatalf("Reasoning() = %q, want %q", got, want)
	}
}
