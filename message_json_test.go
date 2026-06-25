package blades

import (
	"encoding/json"
	"testing"
)

func TestMessageReasoning(t *testing.T) {
	t.Parallel()

	m := &Message{Parts: []Part{
		ReasoningPart{Text: "step1 "},
		TextPart{Text: "answer"},
		ReasoningPart{Text: "step2"},
	}}
	if got, want := m.Reasoning(), "step1 step2"; got != want {
		t.Fatalf("Reasoning() = %q, want %q", got, want)
	}
	// Reasoning content must never leak into the answer.
	if got, want := m.Text(), "answer"; got != want {
		t.Fatalf("Text() = %q, want %q", got, want)
	}
}

func TestMessageJSONRoundTrip(t *testing.T) {
	t.Parallel()

	orig := &Message{
		ID:           "m1",
		Role:         RoleAssistant,
		Author:       "agent",
		InvocationID: "inv1",
		Status:       StatusCompleted,
		FinishReason: "stop",
		TokenUsage:   TokenUsage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
		Actions:      map[string]any{"k": "v"},
		Metadata:     map[string]any{"m": "n"},
		Parts: []Part{
			ReasoningPart{Text: "thinking"},
			TextPart{Text: "hello"},
			FilePart{Name: "f", URI: "u", MIMEType: MIMEImagePNG},
			DataPart{Name: "d", Bytes: []byte("xy"), MIMEType: MIMEAudioMP3},
			ToolPart{ID: "t1", Name: "tool", Request: "{}", Response: "ok", Completed: true},
		},
	}

	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.Reasoning() != "thinking" {
		t.Fatalf("Reasoning() = %q, want %q", got.Reasoning(), "thinking")
	}
	if got.Text() != "hello" {
		t.Fatalf("Text() = %q, want %q", got.Text(), "hello")
	}
	if got.ID != "m1" || got.Role != RoleAssistant || got.InvocationID != "inv1" {
		t.Fatalf("scalar fields not preserved: %+v", got)
	}
	if got.TokenUsage.TotalTokens != 3 {
		t.Fatalf("token usage not preserved: %+v", got.TokenUsage)
	}
	if len(got.Parts) != 5 {
		t.Fatalf("len(Parts) = %d, want 5", len(got.Parts))
	}
	if _, ok := got.Parts[0].(ReasoningPart); !ok {
		t.Fatalf("Parts[0] type = %T, want ReasoningPart", got.Parts[0])
	}
	if fp, ok := got.Parts[2].(FilePart); !ok || fp.MIMEType != MIMEImagePNG {
		t.Fatalf("Parts[2] = %#v, want FilePart png", got.Parts[2])
	}
	if dp, ok := got.Parts[3].(DataPart); !ok || string(dp.Bytes) != "xy" {
		t.Fatalf("Parts[3] = %#v, want DataPart bytes 'xy'", got.Parts[3])
	}
	if tp, ok := got.Parts[4].(ToolPart); !ok || tp.Response != "ok" || !tp.Completed {
		t.Fatalf("Parts[4] = %#v, want completed ToolPart", got.Parts[4])
	}
}

func TestMessageJSONUnknownPartType(t *testing.T) {
	t.Parallel()

	var m Message
	err := json.Unmarshal([]byte(`{"id":"x","parts":[{"type":"bogus"}]}`), &m)
	if err == nil {
		t.Fatal("expected error for unknown part type, got nil")
	}
}
