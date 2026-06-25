package openai

import (
	"strings"

	"github.com/CycleZero/blades"
)

func promptFromMessages(messages []*blades.Message) string {
	var sections []string
	for _, msg := range messages {
		sections = append(sections, msg.Text())
	}
	return strings.Join(sections, "\n")
}
