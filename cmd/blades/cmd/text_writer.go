package cmd

import "github.com/CycleZero/blades/cmd/blades/internal/channel"

type textWriter struct {
	writeText func(string)
}

func (w textWriter) WriteText(chunk string) {
	if w.writeText != nil {
		w.writeText(chunk)
	}
}

func (textWriter) WriteEvent(channel.Event) {}
