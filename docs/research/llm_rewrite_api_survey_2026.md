# Cloud Pro Mode — LLM API 服务调研报告

> 调研日期：2026-03-04
> 适用场景：typeless Cloud Pro Mode — 高频 ASR 文本 rewrite（标点恢复、CSC 纠错、格式化）
> 调研人：AI 云服务解决方案架构师

---

## 1. 需求概述

### 1.1 业务场景

typeless 目前使用端侧 Qwen3-0.6B (INT8) 做 ASR 后处理 rewrite，包括：
- 标点符号恢复
- 常见语音识别错误纠正（CSC）
- 中英文 code-switching 格式化

Cloud Pro Mode 将提供云端 LLM rewrite 作为端侧模型的升级选项，利用更大参数模型获得更好的 rewrite 质量。

### 1.2 核心需求指标

| 维度 | 要求 | 说明 |
|------|------|------|
| **TTFT** | ≤ 400ms | 用户感知延迟，流式输出时首 token 到达时间 |
| **吞吐** | ≥ 100 tok/s | 输出速度需跟上实时语音输入速率 |
| **中文能力** | 优秀 | 标点、CSC、中英混合处理 |
| **指令遵循** | 高 | 不改变原文语义，不添加/删除内容，仅做格式修正 |
| **抗幻觉** | 极高 | ASR rewrite 场景零容忍幻觉——不能凭空加词 |
| **API 兼容** | OpenAI 兼容 | 统一 Swift 网络层，支持多後端切换 |
| **单次调用成本** | < ¥0.001 | 高频调用（~2 次/秒），月成本可控 |

### 1.3 负载估算

```
典型 ASR rewrite 请求：
- Input:  系统 prompt ~200 tok + ASR 文本 ~50 tok = ~250 tok
- Output: rewrite 结果 ~60 tok
- 频率:   每 3-5 秒一次（流式分段）
- 日活:   假设 1000 用户 × 平均 1 小时/天 = ~720K 次/天
- 月 token: ~720K × 310 tok × 30 ≈ 6.7B tok/月
```

---

## 2. 平台横向对比

### 2.1 主要平台概览

#### 2.1.1 Groq (groq.com)

**定位**: LPU (Language Processing Unit) 硬件加速推理，极致低延迟

| 维度 | 详情 |
|------|------|
| **推荐模型** | Llama 3.1 8B Instant / Qwen3 32B |
| **速度** | Llama 3.1 8B: 840 TPS; Qwen3 32B: 662 TPS |
| **价格** | Llama 3.1 8B: $0.05/$0.08 per MTok; Qwen3 32B: $0.29/$0.59 per MTok |
| **TTFT** | ~50-150ms（业界最低之一，LPU 硬件优势） |
| **API 兼容** | 完全 OpenAI 兼容 |
| **中文能力** | Llama 8B 中文一般；Qwen3 32B 中文优秀 |
| **上下文** | 128K |
| **Prompt Cache** | 支持（50% 折扣） |
| **区域** | 美国 |

**优势**: 极致推理速度，TTFT 业界领先；Qwen3 32B 兼顾速度和中文能力
**劣势**: 中国大陆访问需代理；Llama 系列中文能力有限

#### 2.1.2 Cerebras (cerebras.ai)

**定位**: 晶圆级芯片推理，超高吞吐

| 维度 | 详情 |
|------|------|
| **推荐模型** | Llama 3.1 8B / Qwen3 235B (preview) |
| **速度** | Llama 3.1 8B: ~2,200 TPS; Qwen3 235B: ~1,400 TPS |
| **价格** | Llama 3.1 8B: $0.10/$0.10 per MTok; Qwen3 235B: $0.60/$1.20 per MTok |
| **TTFT** | ~100-200ms |
| **API 兼容** | OpenAI 兼容 |
| **中文能力** | Llama 8B 中文一般；Qwen3 235B 中文极佳 |
| **上下文** | 取决于模型 |
| **区域** | 美国 |

**优势**: 吞吐量最高（2,200 TPS），Qwen3 235B 可作为高质量选项
**劣势**: Qwen3 235B 标注为 preview/非生产；中国大陆访问受限

#### 2.1.3 DeepSeek API (platform.deepseek.com)

**定位**: 自研高性价比大模型 API

| 维度 | 详情 |
|------|------|
| **推荐模型** | DeepSeek-V3 |
| **速度** | ~60-100 TPS（官方未公开精确数据） |
| **价格** | 约 ¥1/¥2 per MTok（cache miss）；cache hit 约 ¥0.1/¥2 |
| **TTFT** | ~200-500ms（波动较大，高峰期可达 1s+） |
| **API 兼容** | OpenAI 兼容 |
| **中文能力** | 极佳（原生中文训练） |
| **上下文** | 64K (V3) |
| **区域** | 中国大陆直连 |

**优势**: 中文能力顶级；价格极低（cache 命中后更便宜）；国内直连
**劣势**: 高峰期延迟波动大；可用性历史上有过中断事件；TTFT 不稳定

#### 2.1.4 阿里云百炼 / DashScope (dashscope.aliyuncs.com)

**定位**: 阿里云官方 Qwen 模型服务

| 维度 | 详情 |
|------|------|
| **推荐模型** | Qwen3-Flash / Qwen3-Plus |
| **速度** | Flash ~150-300 TPS（推测）；Plus ~80-150 TPS |
| **价格** | Flash: ¥0.15/¥1.5 per MTok；Plus: ¥0.8/¥2 per MTok |
| **TTFT** | Flash ~100-300ms；Plus ~200-500ms |
| **API 兼容** | OpenAI 兼容模式 |
| **中文能力** | 极佳（Qwen 原生中文） |
| **上下文** | 1M (Flash/Plus) |
| **免费额度** | 各 100 万 token（开通后 90 天内） |
| **区域** | 中国大陆 / 新加坡 / 美国 |

**优势**: 中文最佳（Qwen 亲儿子）；多区域部署；Flash 极便宜；企业级 SLA
**劣势**: 速度比 Groq/Cerebras 慢；国际部署价格较高

#### 2.1.5 Anthropic Claude (api.anthropic.com)

**定位**: 高质量通用大模型

| 维度 | 详情 |
|------|------|
| **推荐模型** | Claude Haiku 4.5 |
| **速度** | ~100-200 TPS（Fastest 档） |
| **价格** | $1.00/$5.00 per MTok |
| **TTFT** | ~200-400ms |
| **API 兼容** | 自有 API（非 OpenAI 兼容，需适配层） |
| **中文能力** | 良好（非原生中文训练，但表现不错） |
| **上下文** | 200K |
| **区域** | 美国 / 欧洲 |

**优势**: 指令遵循最佳；幻觉率极低；输出质量稳定
**劣势**: 价格显著偏高；API 非 OpenAI 兼容需额外适配；中国大陆需代理

**已废弃**: Claude 3 Haiku ($0.25/$1.25) 将于 2026-04-19 退役

#### 2.1.6 SiliconFlow 硅基流动 (siliconflow.cn)

**定位**: 国产模型聚合推理平台，多模型一站接入

| 维度 | 详情 |
|------|------|
| **推荐模型** | Qwen3-8B (免费) / Qwen3-14B (¥0.5/¥2) / DeepSeek-V3.2 (¥2/¥3) |
| **速度** | 7B 级 ~200-400 TPS；32B 级 ~100-200 TPS |
| **价格** | Qwen3-8B 免费; Qwen3-14B ¥0.5/¥2; Qwen3-32B ¥1/¥4 |
| **TTFT** | ~100-400ms（视模型和负载） |
| **API 兼容** | 完全 OpenAI 兼容 |
| **中文能力** | 优秀（托管 Qwen/DeepSeek 原生中文模型） |
| **免费模型** | Qwen3-8B, Qwen3.5-4B, DeepSeek-R1-Distill-7B 等 16+ 款 |
| **区域** | 中国大陆 |

**优势**: 大量免费模型可用；OpenAI 兼容；国内直连；多模型切换零成本
**劣势**: 免费模型有 RPM/TPM 限制；非自建硬件，峰值可能排队

#### 2.1.7 火山引擎 / 豆包 (volcengine.com) — 补充候选

**定位**: 字节跳动云服务，豆包大模型 API

| 维度 | 详情 |
|------|------|
| **推荐模型** | 豆包通用模型 Pro/Lite |
| **价格** | Lite: ¥0.3/¥0.6 per MTok；Pro: ¥0.8/¥2 per MTok（估算） |
| **API 兼容** | OpenAI 兼容模式 |
| **中文能力** | 优秀 |
| **区域** | 中国大陆 |

**优势**: 字节跳动背书；中文能力好；价格有竞争力
**劣势**: API 生态相对较新；文档和社区不如阿里/DeepSeek

---

### 2.2 核心指标对比矩阵

| 平台 | 推荐模型 | TTFT | 吞吐 (TPS) | Input 价格 | Output 价格 | 中文能力 | API 兼容 | 国内直连 |
|------|---------|------|-----------|-----------|------------|---------|---------|---------|
| **Groq** | Qwen3 32B | ~80ms | 662 | $0.29/M | $0.59/M | ★★★★★ | OpenAI | 否 |
| **Groq** | Llama 3.1 8B | ~50ms | 840 | $0.05/M | $0.08/M | ★★★ | OpenAI | 否 |
| **Cerebras** | Llama 3.1 8B | ~100ms | 2,200 | $0.10/M | $0.10/M | ★★★ | OpenAI | 否 |
| **Cerebras** | Qwen3 235B | ~150ms | 1,400 | $0.60/M | $1.20/M | ★★★★★ | OpenAI | 否 |
| **DeepSeek** | V3 | ~300ms | ~80 | ~¥1/M | ~¥2/M | ★★★★★ | OpenAI | 是 |
| **DashScope** | Qwen3-Flash | ~150ms | ~200 | ¥0.15/M | ¥1.5/M | ★★★★★ | OpenAI | 是 |
| **DashScope** | Qwen3-Plus | ~300ms | ~120 | ¥0.8/M | ¥2/M | ★★★★★ | OpenAI | 是 |
| **Claude** | Haiku 4.5 | ~300ms | ~150 | $1.00/M | $5.00/M | ★★★★ | 自有 | 否 |
| **SiliconFlow** | Qwen3-8B | ~200ms | ~300 | 免费 | 免费 | ★★★★ | OpenAI | 是 |
| **SiliconFlow** | Qwen3-32B | ~250ms | ~150 | ¥1/M | ¥4/M | ★★★★★ | OpenAI | 是 |
| **火山引擎** | 豆包 Pro | ~250ms | ~120 | ~¥0.8/M | ~¥2/M | ★★★★★ | OpenAI | 是 |

> 注：TTFT 和 TPS 为估算值，实际表现受负载、地理位置、请求大小等因素影响。
> 价格单位：美元标 $ per MTok，人民币标 ¥ per MTok (每百万 token)。

### 2.3 单次 Rewrite 调用成本估算

典型单次请求：250 input tokens + 60 output tokens

| 平台 | 模型 | 单次成本 | 月成本 (720K 次) | 换算 (¥) |
|------|------|---------|----------------|---------|
| **Groq** | Llama 3.1 8B | $0.000017 | $12.6 | ¥92 |
| **Groq** | Qwen3 32B | $0.000108 | $77.8 | ¥567 |
| **Cerebras** | Llama 3.1 8B | $0.000031 | $22.3 | ¥163 |
| **DashScope** | Qwen3-Flash | ¥0.000128 | ¥92 | ¥92 |
| **DashScope** | Qwen3-Plus | ¥0.000320 | ¥230 | ¥230 |
| **DeepSeek** | V3 | ¥0.000370 | ¥266 | ¥266 |
| **SiliconFlow** | Qwen3-8B | 免费 | 免费 | 免费 |
| **SiliconFlow** | Qwen3-32B | ¥0.000490 | ¥353 | ¥353 |
| **Claude** | Haiku 4.5 | $0.000550 | $396 | ¥2,890 |

> 汇率按 $1 = ¥7.3 估算。

---

## 3. ASR Rewrite 抗幻觉评估

### 3.1 评估方法

ASR rewrite 对「幻觉」零容忍，模型必须满足：
1. **不添词**: 不能凭空增加原文没有的内容
2. **不删词**: 不能遗漏原文的任何语义内容
3. **不改义**: 仅做标点/格式/CSC 修正，不能改变原文含义
4. **稳定输出**: 相同输入多次调用结果一致

### 3.2 各模型抗幻觉能力评估

| 模型类别 | 抗幻觉能力 | 说明 |
|---------|-----------|------|
| **Qwen3 系列 (≥14B)** | ★★★★★ | 中文 ASR 文本理解最佳，指令遵循严格，极少产生额外内容 |
| **Qwen3-8B** | ★★★★ | 中文能力好，偶尔在复杂 code-switching 场景过度修正 |
| **Claude Haiku 4.5** | ★★★★★ | 指令遵循最严格，几乎不产生幻觉，但中文不如 Qwen 原生 |
| **DeepSeek V3** | ★★★★ | 中文能力强，但作为通用模型偶有"创造性"输出 |
| **Llama 3.1 8B** | ★★★ | 英文指令遵循好，中文场景容易过度翻译/改写 |
| **豆包 Pro** | ★★★★ | 中文好，但偶有"补全"倾向 |

### 3.3 Prompt 设计建议

ASR rewrite 场景推荐使用如下 system prompt 策略：

```
你是一个语音识别文本后处理助手。
规则：
1. 仅添加/修正标点符号
2. 仅修正明显的语音识别错误（同音字替换）
3. 严禁添加原文没有的内容
4. 严禁删除原文的任何内容
5. 严禁改变原文的语义
6. 如果原文没有问题，原样输出

输入：{asr_text}
输出：
```

**关键点**:
- 使用 `temperature=0` 确保输出稳定
- 添加 few-shot 示例提升指令遵循
- 对于 OpenAI 兼容 API，可设置 `top_p=1, frequency_penalty=0` 避免创造性输出

---

## 4. 技术架构推荐

### 4.1 推荐方案分层

```
┌─────────────────────────────────────────────┐
│              tipless Cloud Pro Mode          │
├─────────────────────────────────────────────┤
│  Tier 0 (免费)    │ SiliconFlow Qwen3-8B    │  ← 免费额度内
│  Tier 1 (标准)    │ DashScope Qwen3-Flash   │  ← 性价比最优
│  Tier 2 (高级)    │ Groq Qwen3 32B          │  ← 速度+质量
│  Tier 3 (极速)    │ Groq Llama 8B           │  ← 极致低延迟
│  Fallback         │ DeepSeek V3             │  ← 备用
└─────────────────────────────────────────────┘
```

### 4.2 推荐首选方案

**首选: 阿里云 DashScope Qwen3-Flash**

理由：
1. **中文能力最强** — Qwen 系列原生中文训练，ASR rewrite 场景最优
2. **价格极低** — ¥0.15/¥1.5 per MTok，月成本约 ¥92
3. **速度够用** — TTFT ~150ms，吞吐 ~200 TPS，满足实时要求
4. **国内直连** — 无需代理，延迟稳定
5. **OpenAI 兼容** — `base_url` 切换即可
6. **企业级 SLA** — 阿里云背书，99.9% 可用性
7. **Flash 模型定位** — 专为「简单任务、快速响应」设计，完美匹配 ASR rewrite

**备选: Groq Qwen3 32B**（面向海外用户或追求极致速度）

### 4.3 Swift 网络层架构设计

```swift
// MARK: - Cloud Rewrite API Client

/// OpenAI-compatible API configuration
struct CloudRewriteConfig {
    let baseURL: String        // e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
    let apiKey: String
    let model: String          // e.g. "qwen3-flash"
    let maxTokens: Int         // = 100 (rewrite output typically < 80 tok)
    let temperature: Double    // = 0.0 (deterministic for ASR rewrite)
    let timeoutSeconds: Double // = 5.0

    /// 预置配置
    static let dashscopeFlash = CloudRewriteConfig(
        baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        apiKey: "sk-xxx", model: "qwen3-flash",
        maxTokens: 100, temperature: 0.0, timeoutSeconds: 5.0
    )

    static let groqQwen3 = CloudRewriteConfig(
        baseURL: "https://api.groq.com/openai/v1",
        apiKey: "gsk-xxx", model: "qwen3-32b",
        maxTokens: 100, temperature: 0.0, timeoutSeconds: 5.0
    )

    static let siliconflowFree = CloudRewriteConfig(
        baseURL: "https://api.siliconflow.cn/v1",
        apiKey: "sk-xxx", model: "Qwen/Qwen3-8B",
        maxTokens: 100, temperature: 0.0, timeoutSeconds: 5.0
    )
}

/// Unified cloud rewrite client using OpenAI-compatible chat/completions
actor CloudRewriteClient {
    private let config: CloudRewriteConfig
    private let session: URLSession

    init(config: CloudRewriteConfig) {
        self.config = config
        let urlConfig = URLSessionConfiguration.default
        urlConfig.timeoutIntervalForRequest = config.timeoutSeconds
        self.session = URLSession(configuration: urlConfig)
    }

    /// Non-streaming rewrite (preferred for short outputs)
    func rewrite(_ asrText: String) async throws -> String {
        let url = URL(string: "\(config.baseURL)/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.maxTokens,
            "messages": [
                ["role": "system", "content": Self.systemPrompt],
                ["role": "user", "content": asrText]
            ]
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)
        // Parse OpenAI-compatible response
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let choices = json["choices"] as! [[String: Any]]
        let message = choices[0]["message"] as! [String: Any]
        return message["content"] as! String
    }

    /// Streaming rewrite using SSE (for real-time display)
    func rewriteStream(_ asrText: String) -> AsyncThrowingStream<String, Error> {
        // SSE streaming implementation using URLSession bytes
        // All OpenAI-compatible APIs use the same SSE format:
        // data: {"choices":[{"delta":{"content":"..."}}]}
        // ...
    }

    private static let systemPrompt = """
    你是一个语音识别文本后处理助手。
    规则：
    1. 仅添加/修正标点符号
    2. 仅修正明显的语音识别错误（同音字替换）
    3. 严禁添加原文没有的内容
    4. 严禁删除原文的任何内容
    5. 严禁改变原文的语义
    6. 如果原文没有问题，原样输出
    """
}
```

### 4.4 多后端 Failover 策略

```
请求 → DashScope Flash (主)
        ├── 成功 → 返回
        └── 超时/失败 → SiliconFlow Qwen3-8B (备)
                         ├── 成功 → 返回
                         └── 超时/失败 → 本地 Qwen3-0.6B (兜底)
```

关键设计点：
1. **超时控制**: 每个后端 5s 超时，总超时 10s
2. **熔断器**: 连续 3 次失败后自动切换到下一个后端，60s 后重试
3. **本地兜底**: 始终保留端侧 Qwen3-0.6B 作为最终 fallback
4. **延迟监控**: 记录每次 TTFT，动态调整后端优先级

---

## 5. 平台详细评分

### 5.1 综合评分（满分 5 分）

| 平台 | 速度 | 价格 | 中文 | 指令遵循 | 稳定性 | API 兼容 | 国内可用 | 综合 |
|------|------|------|------|---------|--------|---------|---------|------|
| **DashScope Flash** | 4 | 5 | 5 | 4 | 5 | 4 | 5 | **4.6** |
| **Groq Qwen3 32B** | 5 | 4 | 5 | 4 | 4 | 5 | 2 | **4.1** |
| **SiliconFlow Qwen3-8B** | 4 | 5 | 4 | 3 | 3 | 5 | 5 | **4.1** |
| **DeepSeek V3** | 3 | 4 | 5 | 4 | 3 | 5 | 5 | **4.0** |
| **Cerebras Llama 8B** | 5 | 5 | 2 | 3 | 4 | 5 | 2 | **3.7** |
| **Claude Haiku 4.5** | 3 | 2 | 4 | 5 | 5 | 2 | 1 | **3.1** |

> 权重：中文 × 1.5, 价格 × 1.3, 速度 × 1.2, 国内可用 × 1.2, 其他 × 1.0

### 5.2 推荐优先级

| 优先级 | 方案 | 适用场景 | 月成本 |
|-------|------|---------|-------|
| **P0** | DashScope Qwen3-Flash | 国内用户默认方案 | ~¥92 |
| **P1** | SiliconFlow Qwen3-8B | 开发测试 / 免费体验 | ¥0 |
| **P2** | Groq Qwen3 32B | 海外用户 / 极致速度 | ~$78 |
| **P3** | DeepSeek V3 | 备选后端 | ~¥266 |

---

## 6. 风险与注意事项

### 6.1 安全

- API Key 必须存储在 Keychain 中，不能硬编码或存储在 UserDefaults
- 传输层使用 HTTPS + Certificate Pinning（可选）
- 用户 ASR 文本经过云端处理，需在隐私政策中说明

### 6.2 成本控制

- 实现 token 计数本地预估，超出阈值自动降级到本地模型
- 设置用户每日调用上限（免费用户 100 次/天，付费用户不限）
- 使用 DashScope Batch API 做离线批量处理（半价）

### 6.3 可用性

- DeepSeek 历史上有过服务中断事件，不建议作为唯一后端
- SiliconFlow 免费模型有 RPM 限制（~1000 RPM），高并发场景需付费
- Groq 免费 tier 有严格速率限制，生产环境需 Developer tier ($10 起)

### 6.4 模型版本管理

- DashScope 模型存在版本迭代（如 qwen3-flash 会更新底层模型）
- 建议使用固定版本号（如 `qwen3-max-2026-01-23`）避免行为变更
- 定期评估新版本效果，手动从开发环境验收后切换

---

## 7. 实施路线图

### Phase 1: MVP
- 接入 DashScope Qwen3-Flash 作为唯一云端后端
- 实现 OpenAI 兼容 `CloudRewriteClient`
- 添加 Settings 页面：Cloud Pro Mode 开关 + API Key 输入
- 本地 Qwen3-0.6B 作为 fallback

### Phase 2: 多后端
- 添加 Groq / SiliconFlow 作为可选后端
- 实现自动 failover + 熔断器
- 添加延迟监控和后端健康度追踪

### Phase 3: 优化
- 实现 Prompt Caching（DashScope / Groq 均支持）
- 添加本地 token 计数和成本仪表盘
- 支持用户自定义 API endpoint（兼容任何 OpenAI API 服务）

---

## 8. 结论

对于 typeless Cloud Pro Mode 的 ASR rewrite 场景：

1. **首选 DashScope Qwen3-Flash** — 中文最强、价格最低、国内直连、速度达标
2. **SiliconFlow 免费模型可作为入门体验** — 零成本让用户试用 Cloud Pro
3. **Groq 是海外用户的最佳选择** — 极致速度 + Qwen3 32B 高质量
4. **Claude Haiku 虽然指令遵循最好，但性价比不适合高频 rewrite 场景**
5. **所有 OpenAI 兼容后端共享同一个 Swift 网络层**，切换成本为零

---

## 参考链接

- [Groq Pricing](https://groq.com/pricing/)
- [Cerebras Pricing](https://cerebras.ai/pricing)
- [DeepSeek API Docs](https://api-docs.deepseek.com/)
- [阿里云百炼模型列表](https://help.aliyun.com/zh/model-studio/getting-started/models)
- [Anthropic Claude Models](https://platform.claude.com/docs/en/docs/about-claude/models)
- [SiliconFlow Pricing](https://siliconflow.cn/pricing)
- [火山引擎模型服务](https://www.volcengine.com/docs/82379/1099320)
