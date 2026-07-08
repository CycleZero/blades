// Command deepseek-long-conversation tests deepseek v4 reasoning_content passthrough
// over a long (>50 message) multi-invocation conversation with tool calling.
//
// It simulates a real agent scenario: multiple user questions are interleaved,
// each triggering tool calls, and every model request includes the full
// accumulated session history via WithContext(true).  This is exactly the
// pattern that triggers the "reasoning_content must be passed back" 400 error
// if the outgoing assistant messages are missing reasoning.
//
// Run (key via env, never stored):
//
//	DEEPSEEK_API_KEY=sk-... go run ./deepseek-long-conversation
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/CycleZero/blades"
	"github.com/CycleZero/blades/contrib/openai"
	"github.com/CycleZero/blades/tools"
)

// -------------------------------------------------------------------------
// tools
// -------------------------------------------------------------------------

type PopReq struct {
	City string `json:"city" jsonschema:"中文城市名"`
}
type PopRes struct {
	City       string `json:"city"`
	Population int    `json:"population"`
}

var populations = map[string]int{
	"北京": 21893000, "上海": 24870000, "广州": 18676000,
	"深圳": 17560000, "成都": 21268000, "杭州": 12204000,
	"武汉": 13778900, "西安": 13130000,
}

func getPopulation(ctx context.Context, req PopReq) (PopRes, error) {
	city := strings.TrimSpace(req.City)
	p, ok := populations[city]
	if !ok {
		return PopRes{City: city, Population: 0}, nil
	}
	return PopRes{City: city, Population: p}, nil
}

type AddReq struct {
	A int `json:"a"`
	B int `json:"b"`
}
type AddRes struct {
	Sum int `json:"sum"`
}

func add(ctx context.Context, req AddReq) (AddRes, error) {
	return AddRes{Sum: req.A + req.B}, nil
}

// -------------------------------------------------------------------------
// logging session wrapper — prints every Append so we see the full flow
// -------------------------------------------------------------------------

type logSession struct {
	blades.Session
}

func (s *logSession) Append(ctx context.Context, m *blades.Message) error {
	tools := 0
	for _, p := range m.Parts {
		if _, ok := p.(blades.ToolPart); ok {
			tools++
		}
	}
	rl := len([]rune(m.Reasoning()))
	txt := trunc(m.Text(), 60)
	logf("   APPEND  role=%-9s status=%-10s reasoning=%-4d字 tools=%d text=%q",
		m.Role, m.Status, rl, tools, txt)
	return s.Session.Append(ctx, m)
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

func main() {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		fmt.Println("FATAL: DEEPSEEK_API_KEY not set")
		os.Exit(1)
	}
	baseURL := env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
	model := env("DEEPSEEK_MODEL", "deepseek-v4-flash")

	banner("CONFIG")
	logf("base_url=%s  model=%s  reasoning_effort=xhigh", baseURL, model)

	popTool, err := tools.NewFunc("get_city_population", "查询指定中文城市的常住人口", getPopulation)
	must(err)
	addTool, err := tools.NewFunc("add", "计算两个整数之和(一次只能加两个数)", add)
	must(err)

	mp := openai.NewModel(model, openai.Config{
		APIKey:          apiKey,
		BaseURL:         baseURL,
		ReasoningEffort: "xhigh",
	})

	agent, err := blades.NewAgent(
		"LongConvAgent",
		blades.WithModel(mp),
		blades.WithInstruction("你是严谨的数据助手。所有数值必须通过工具查询或计算得到,禁止凭空编造。add一次只能加两个数。"),
		blades.WithTools(popTool, addTool),
		blades.WithContext(true), // ← 每次 model 请求加载全量 session 历史
		blades.WithMaxIterations(15),
	)
	must(err)

	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
	defer cancel()

	baseSession := blades.NewSession()
	session := &logSession{Session: baseSession}
	runner := blades.NewRunner(agent)

	questions := []string{
		"查询北京和上海的人口。",
		"把北京和上海的人口相加,告诉我总和。",
		"查询广州的人口,把它也加到之前的总和里。",
		"深圳的人口是多少？把深圳也加入总和。",
		"查询成都的人口,一并加到总和里。",
		"把杭州和武汉的人口查出来,然后相加。",
		"把刚才杭州+武汉的结果加到我们一直维护的总和里。",
		"西安的人口是多少？加入总和。",
		"杭州和西安的人口哪个更大？",
		"查询一个不在你数据库里的城市南京,看看会发生什么。",
		"现在告诉我:历史上一共查询了哪些城市,各自人口多少,最终总和是多少。",
		"把成都和武汉的人口用add工具相加。",
		"如果北京人口增长5%变成22987650,从0开始重新累加上海、广州、深圳、成都的人口,再加上北京新值,告诉我每一步的结果。",
		"告诉我:在全部对话中,add工具一共被调用了多少次？",
		"最后,请列出本次对话中所有你查询过的城市(不含不存在的城市),按人口从大到小排序。",
	}

	for i, q := range questions {
		banner(fmt.Sprintf("INVOCATION #%d", i+1))
		logf("QUESTION: %s", q)

		out, err := runner.Run(ctx, blades.UserMessage(q), blades.WithSession(session))
		if err != nil {
			logf("FATAL: invocation failed — %v", err)
			os.Exit(1)
		}
		logf("ANSWER: %s", trunc(out.Text(), 200))
		logf("TOKENS: in=%d out=%d total=%d", out.TokenUsage.InputTokens, out.TokenUsage.OutputTokens, out.TokenUsage.TotalTokens)

		hist, _ := session.History(ctx)
		logf("HISTORY: %d messages so far", len(hist))
		fmt.Println()
	}

	banner("VERIFICATION")

	hist, err := session.History(ctx)
	must(err)
	logf("total messages in session = %d", len(hist))

	totalR, missingR, totalA := 0, 0, 0
	for i, m := range hist {
		rl := len([]rune(m.Reasoning()))
		totalR += rl
		if m.Role == blades.RoleAssistant || m.Role == blades.RoleTool {
			totalA++
			if rl == 0 {
				missingR++
				logf("WARN: assistant/tool message #%d has NO reasoning (text=%q)", i, trunc(m.Text(), 40))
			}
		}
	}
	logf("assistant+tool messages = %d  (with reasoning = %d  missing reasoning = %d)",
		totalA, totalA-missingR, missingR)
	logf("total reasoning characters = %d", totalR)

	if len(hist) < 50 {
		logf("NOTE: history has %d messages, target was >50; consider more invocations.", len(hist))
	} else {
		logf("OK: >50 messages in session history, long conversation validated.")
	}

	if missingR > 0 {
		logf("ISSUE: %d assistant/tool messages lack reasoning_content.", missingR)
	} else {
		logf("OK: all assistant/tool messages carry reasoning_content.")
	}

	banner("DONE")
}

// -------------------------------------------------------------------------
// helpers
// -------------------------------------------------------------------------

func ts() string { return time.Now().Format("15:04:05") }

func logf(format string, a ...any) {
	fmt.Printf("%s | %s\n", ts(), fmt.Sprintf(format, a...))
}

func banner(title string) {
	fmt.Printf("\n%s | ====== %s ======\n", ts(), title)
}

func must(err error) {
	if err != nil {
		logf("FATAL: %v", err)
		os.Exit(1)
	}
}

func env(key, dflt string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return dflt
}

func trunc(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
