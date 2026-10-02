// body.go 构造 agent_chat_generation 请求体（纯透传模式）。
// 移植自 qoderwork2api internal/upstream/body.go：客户端消息全量转发，
// tools 仅在客户端显式传入时注入。
package qoder

import (
	"encoding/json"
	"time"
)

// buildAgentBody 构造请求体。
//   - messages：客户端原始消息列表（可含 system/assistant/tool 多轮）
//   - modelKey：上游模型 key（如 dmodel）
//   - mc：该模型的上游条目（可以为 nil，如走静态表兜底）；format/source 取自它
//   - tools：客户端传来的 OpenAI tools 数组；为空则不注入 tools 字段
//   - enableReasoning：是否启用思考模式
//   - reasoningEffort：思考强度 low/medium/high，空串表示不传
//   - contextWindow：选定的上下文档位；<=0 时不注入 context_length/parameters
//
// 注意：developer 角色必须改写为 system。
func buildAgentBody(messages []map[string]any, modelKey string, mc *DynamicModel, tools []any, enableReasoning bool, reasoningEffort string, contextWindow int64) ([]byte, error) {
	// developer → system（浅拷贝消息避免污染调用方数据）
	msgs := make([]map[string]any, len(messages))
	for i, m := range messages {
		cp := make(map[string]any, len(m)+1)
		for k, v := range m {
			cp[k] = v
		}
		if r, _ := cp["role"].(string); r == "developer" {
			cp["role"] = "system"
		}
		msgs[i] = cp
	}

	// 最后一条 user 消息文本（chat_context.text 上游协议要求必填）
	prompt := ""
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i]["role"] == "user" {
			if c, ok := msgs[i]["content"].(string); ok && c != "" {
				prompt = c
				break
			}
		}
	}

	now := time.Now()
	newUUID := uuid4()

	// format/source 取上游真值，未下发（或走静态表兜底）时才用兜底常量。
	// ⚠️ model_config.source 即思考总开关：旧实现完全不下发该字段（issue #32 思考过程不暴露）。
	format := defaultFormat
	source := defaultSource
	maxIn := int64(180000)
	if mc != nil {
		if mc.Format != "" {
			format = mc.Format
		}
		if mc.Source != "" {
			source = mc.Source
		}
		if mc.MaxInputTokens > 0 {
			maxIn = mc.MaxInputTokens
		}
	}

	// 上下文档位透传（issue #27）：对齐 9router——parameters.context_length + model_config.max_input_tokens。
	modelCfg := map[string]any{
		"key": modelKey, "is_reasoning": enableReasoning,
		"format": format, "source": source,
		"max_input_tokens": maxIn,
	}
	params := map[string]any{}
	if contextWindow > 0 {
		params["context_length"] = contextWindow
		modelCfg["max_input_tokens"] = contextWindow
	}
	if enableReasoning && reasoningEffort != "" {
		params["reasoning_effort"] = reasoningEffort
	}
	if len(params) == 0 {
		params = nil
	}
	base := map[string]any{
		"request_id":       newUUID,
		"chat_record_id":   newUUID,
		"request_set_id":   uuid4(),
		"session_id":       uuid4(),
		"stream":           true,
		"aliyun_user_type": "personal_professional_trial",
		"agent_id":         "agent_common",
		"chat_task":        "FREE_INPUT",
		"is_reply":         true,
		"image_urls":       nil,
		"session_type":     "qodercli",
		"model_config":     modelCfg,
		"chat_context": map[string]any{
			"chatPrompt": "",
			"text":       map[string]any{"type": "text", "text": prompt},
			"extra": map[string]any{
				"context":         []any{},
				"modelConfig":     map[string]any{"key": modelKey, "is_reasoning": enableReasoning},
				"originalContent": map[string]any{"type": "text", "text": prompt},
			},
			"features":  []any{},
			"imageUrls": nil,
		},
		"messages": msgs,
		"business": map[string]any{
			"id":       uuid4(),
			"begin_at": now.UnixMilli(),
			"name":     truncateRunes(prompt, 30),
		},
	}
	if params != nil {
		base["parameters"] = params
	}

	if len(tools) > 0 {
		base["tools"] = tools
	}

	return json.Marshal(base)
}

// truncateRunes 截断到 n 个 rune。
func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// resolveContextWindow 解析本次请求应使用的上下文档位。
// 客户端值校验对齐官方 TZ；默认档与非法值落点取 max(AvailableWindows)
// （宁高勿低，对齐 workbuddy2api 广告口径），而非官方 FHA 的 is_default：
//  1. requested>0 且校验通过 → 用 requested
//  2. 否则 → max(AvailableWindows)（有档位表时）
//  3. 否则 → ContextWindow → MaxInputTokens → 0（不注入）
func resolveContextWindow(requested int64, mc *DynamicModel) int64 {
	if requested > 0 {
		if mc == nil {
			return requested
		}
		if len(mc.AvailableWindows) > 0 {
			for _, w := range mc.AvailableWindows {
				if w == requested {
					return requested
				}
			}
		} else {
			if mc.MaxInputTokens <= 0 || requested <= mc.MaxInputTokens {
				return requested
			}
		}
	}
	if mc == nil {
		return 0
	}
	if n := len(mc.AvailableWindows); n > 0 {
		return mc.AvailableWindows[n-1] // 升序，取最大档
	}
	if mc.ContextWindow > 0 {
		return mc.ContextWindow
	}
	if mc.MaxInputTokens > 0 {
		return mc.MaxInputTokens
	}
	return 0
}

// parseContextWindowHint 从 OpenAI 请求体读取客户端上下文档位提示。
// 同时认 context_length（上游参数名）与 context_window（官方 SDK 驼峰名）。
func parseContextWindowHint(body []byte) int64 {
	var hint struct {
		ContextLength int64 `json:"context_length"`
		ContextWindow int64 `json:"context_window"`
	}
	if err := json.Unmarshal(body, &hint); err != nil {
		return 0
	}
	if hint.ContextLength > 0 {
		return hint.ContextLength
	}
	return hint.ContextWindow
}
