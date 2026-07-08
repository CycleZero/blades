// Command deepseek-reasoning-tools is a detailed, end-to-end test of multi-turn
// tool calling with reasoning-content (思考链) passthrough, against a DeepSeek
// (OpenAI-compatible) endpoint.
//
// It exercises:
//   - streaming + multi-turn tool calling (population lookups -> stepwise add)
//   - reasoning_content passthrough via Message.Reasoning()
//   - the full reasoning chain persisted into the session history
//   - JSON serialize/deserialize round-trip of the history (type-tagged Parts),
//     i.e. exactly what a custom persistable Session would do.
//
// Configuration is read from the environment (no secrets in source):
//
//	DEEPSEEK_API_KEY   required
//	DEEPSEEK_BASE_URL  optional, default https://api.deepseek.com
//	DEEPSEEK_MODEL     optional, default deepseek-v4-flash
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/CycleZero/blades"
	"github.com/CycleZero/blades/contrib/openai"
	"github.com/CycleZero/blades/tools"
)

// ---------------------------------------------------------------------------
// tool 1: city population lookup
// ---------------------------------------------------------------------------

type PopReq struct {
	City string `json:"city" jsonschema:"中文城市名，例如 北京"`
}
type PopRes struct {
	City       string `json:"city"`
	Population int    `json:"population" jsonschema:"该城市常住人口（整数）"`
}

var populations = map[string]int{
	"北京": 21893000,
	"上海": 24870000,
	"广州": 18676000,
	"深圳": 17560000,
}

func getPopulation(ctx context.Context, req PopReq) (PopRes, error) {
	logf("🔧 TOOL  get_city_population(city=%q) 被调用", req.City)
	p, ok := populations[strings.TrimSpace(req.City)]
	if !ok {
		logf("        ↳ 未找到该城市，返回 0")
		return PopRes{City: req.City, Population: 0}, nil
	}
	logf("        ↳ population = %d", p)
	return PopRes{City: req.City, Population: p}, nil
}

// ---------------------------------------------------------------------------
// tool 2: binary add (forces multi-step chaining for 3 numbers)
// ---------------------------------------------------------------------------

type AddReq struct {
	A int `json:"a" jsonschema:"第一个加数"`
	B int `json:"b" jsonschema:"第二个加数"`
}
type AddRes struct {
	Sum int `json:"sum"`
}

func add(ctx context.Context, req AddReq) (AddRes, error) {
	logf("🔧 TOOL  add(a=%d, b=%d) 被调用", req.A, req.B)
	s := req.A + req.B
	logf("        ↳ sum = %d", s)
	return AddRes{Sum: s}, nil
}

// ---------------------------------------------------------------------------

func main() {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		logf("FATAL: 环境变量 DEEPSEEK_API_KEY 未设置")
		os.Exit(1)
	}
	baseURL := envDefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
	model := envDefault("DEEPSEEK_MODEL", "deepseek-v4-flash")

	banner("配置 / CONFIG")
	logf("base_url          = %s", baseURL)
	logf("model             = %s", model)
	logf("reasoning_effort  = xhigh (max)")
	logf("stream            = true")
	logf("max_iterations    = 12")

	popTool, err := tools.NewFunc(
		"get_city_population",
		"查询指定中文城市的常住人口，返回整数人口数。",
		getPopulation,
	)
	must(err)
	addTool, err := tools.NewFunc(
		"add",
		"计算两个整数相加的和（一次只能加两个数）。",
		add,
	)
	must(err)

	mp := openai.NewModel(model, openai.Config{
		APIKey:          apiKey,
		BaseURL:         baseURL,
		ReasoningEffort: "xhigh", // 思考强度拉满
	})

	agent, err := blades.NewAgent(
		"DeepSeek Reasoning Agent",
		blades.WithModel(mp),
		blades.WithInstruction("你是一个严谨的助手。必须使用提供的工具完成数值查询与计算，禁止凭空编造数字。add 工具一次只能相加两个整数，多个数请分步调用 add 逐步累加。"),
		blades.WithTools(popTool, addTool),
		blades.WithMaxIterations(12),
	)
	must(err)

	question := "请用工具分别查询【北京】【上海】【广州】三个城市的人口，然后用 add 工具把它们逐步相加，最后明确告诉我这三座城市的总人口是多少。"
	banner("用户问题 / USER")
	logf("%s", question)

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	session := blades.NewSession()
	runner := blades.NewRunner(agent)

	banner("流式执行（多轮工具调用 + 思考链回传）/ STREAMING")
	var (
		reasoningTurn strings.Builder
		answer        strings.Builder
		toolRounds    int
		lastKind      string
	)
	for msg, err := range runner.RunStream(ctx, blades.UserMessage(question), blades.WithSession(session)) {
		if err != nil {
			fmt.Println()
			logf("FATAL: 流式调用出错: %v", err)
			os.Exit(1)
		}
		// blades streams BOTH incremental deltas (Incomplete) and a final
		// consolidated message (Completed) per turn. Accumulate/print content
		// from the deltas only, to avoid double-counting the consolidation.
		isDelta := msg.Status != blades.StatusCompleted
		if r := msg.Reasoning(); r != "" && isDelta {
			reasoningTurn.WriteString(r)
			if lastKind != "reasoning" {
				fmt.Printf("\n%s | 🧠 思考> ", ts())
				lastKind = "reasoning"
			}
			fmt.Print(r)
		}
		if t := msg.Text(); t != "" && isDelta {
			answer.WriteString(t)
			if lastKind != "text" {
				fmt.Printf("\n%s | 💬 回答> ", ts())
				lastKind = "text"
			}
			fmt.Print(t)
		}
		// executed tool round (含调用参数与返回结果)
		if msg.Role == blades.RoleTool && msg.Status == blades.StatusCompleted {
			toolRounds++
			fmt.Println()
			logf("──────── 工具轮 #%d / TOOL ROUND ────────", toolRounds)
			for _, p := range msg.Parts {
				if tp, ok := p.(blades.ToolPart); ok {
					logf("   ▶ %s(%s)  ⇒  %s", tp.Name, compact(tp.Request), compact(tp.Response))
				}
			}
			logf("   本轮思考累计 %d 字", runeLen(reasoningTurn.String()))
			reasoningTurn.Reset()
			answer.Reset() // 工具轮后开始新的助手回合，清空已打印的前导文本
			lastKind = ""
		}
		// final assistant message of a turn
		if msg.Role == blades.RoleAssistant && msg.Status == blades.StatusCompleted {
			fmt.Println()
			logf("──────── 助手消息完成 / ASSISTANT COMPLETED ────────")
			logf("   tokens: in=%d out=%d total=%d",
				msg.TokenUsage.InputTokens, msg.TokenUsage.OutputTokens, msg.TokenUsage.TotalTokens)
			lastKind = ""
		}
	}

	banner("最终答案 / FINAL ANSWER")
	logf("%s", strings.TrimSpace(answer.String()))

	// ---- 会话历史 + 完整思考链 ----
	banner("会话历史 / SESSION HISTORY (含思考链)")
	hist, err := session.History(ctx)
	must(err)
	totalReasoning := 0
	for i, m := range hist {
		rl := runeLen(m.Reasoning())
		totalReasoning += rl
		logf("[%d] role=%-9s author=%-24s status=%-9s tools=%d reasoning=%d字 text=%q",
			i, m.Role, m.Author, m.Status, countToolParts(m), rl, truncate(m.Text(), 60))
	}
	logf("整条思考链合计 %d 字，跨 %d 条消息", totalReasoning, len(hist))

	// ---- JSON 序列化/反序列化 round-trip（自定义可持久化 Session 的核心）----
	banner("历史序列化往返 / JSON ROUND-TRIP")
	data, err := json.MarshalIndent(hist, "", "  ")
	must(err)
	hasReasoningTag := strings.Contains(string(data), `"type": "reasoning"`)
	logf("序列化字节数 = %d", len(data))
	logf(`JSON 中是否包含 "type": "reasoning" 标签 = %v`, hasReasoningTag)

	var restored []*blades.Message
	must(json.Unmarshal(data, &restored))
	afterReasoning := 0
	for _, m := range restored {
		afterReasoning += runeLen(m.Reasoning())
	}
	logf("反序列化恢复 %d 条消息", len(restored))
	logf("思考链字数  序列化前=%d  反序列化后=%d  完整保留=%v",
		totalReasoning, afterReasoning, totalReasoning == afterReasoning)

	banner("测试完成 / DONE")
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func ts() string { return time.Now().Format("15:04:05.000") }

func logf(format string, a ...any) {
	fmt.Printf("%s | %s\n", ts(), fmt.Sprintf(format, a...))
}

func banner(title string) {
	fmt.Println()
	fmt.Printf("%s | ==================== %s ====================\n", ts(), title)
}

func must(err error) {
	if err != nil {
		logf("FATAL: %v", err)
		os.Exit(1)
	}
}

func envDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func compact(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	return truncate(s, 200)
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func runeLen(s string) int { return len([]rune(s)) }

func countToolParts(m *blades.Message) int {
	n := 0
	for _, p := range m.Parts {
		if _, ok := p.(blades.ToolPart); ok {
			n++
		}
	}
	return n
}
