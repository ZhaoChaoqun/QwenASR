# Qwen3-0.6B

> 翻译自 https://huggingface.co/Qwen/Qwen3-0.6B

## Qwen3 亮点

Qwen3 是 Qwen 系列大语言模型的最新一代，提供了完整的稠密模型和混合专家（MoE）模型套件。基于大规模训练，Qwen3 在推理、指令遵循、Agent 能力和多语言支持方面取得了突破性进展，主要特性如下：

- **在单一模型中独创性地支持思考模式**（用于复杂逻辑推理、数学和编程）和**非思考模式**（用于高效的通用对话）**之间的无缝切换**，确保在各种场景下都能达到最优性能。
- **推理能力显著增强**，在数学、代码生成和常识逻辑推理方面超越了此前的 QwQ（思考模式）和 Qwen2.5 Instruct 模型（非思考模式）。
- **卓越的人类偏好对齐**，在创意写作、角色扮演、多轮对话和指令遵循方面表现优异，提供更自然、引人入胜和沉浸式的对话体验。
- **专精的 Agent 能力**，能在思考和非思考模式下精确集成外部工具，在复杂的 Agent 任务中达到开源模型的领先性能。
- **支持 100 多种语言和方言**，在**多语言指令遵循**和**翻译**方面具有强大能力。

## 模型概览

**Qwen3-0.6B** 具有以下特征：

- 类型：因果语言模型
- 训练阶段：预训练 & 后训练
- 参数量：0.6B
- 参数量（不含 Embedding）：0.44B
- 层数：28
- 注意力头数（GQA）：Q 头 16 个，KV 头 8 个
- 上下文长度：32,768

更多详情（包括基准评测、硬件要求和推理性能），请参考我们的 [博客](https://qwenlm.github.io/blog/qwen3/)、[GitHub](https://github.com/QwenLM/Qwen3) 和 [文档](https://qwen.readthedocs.io/en/latest/)。

> 如果遇到严重的无限重复问题，请参考[最佳实践](#最佳实践)部分的最优采样参数，并将 `presence_penalty` 设置为 1.5。

## 快速开始

Qwen3 的代码已包含在最新版 Hugging Face `transformers` 中，建议使用最新版本的 `transformers`。

使用 `transformers<4.51.0` 时会遇到以下错误：

```
KeyError: 'qwen3'
```

以下代码示例展示了如何使用模型根据给定输入生成内容。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 准备模型输入
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 在思考模式和非思考模式之间切换。默认为 True。
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 执行文本补全
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 解析思考内容
try:
    # rindex 查找 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

部署方面，你可以使用 `sglang>=0.4.6.post1` 或 `vllm>=0.8.5` 创建兼容 OpenAI 的 API 端点：

- SGLang：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3
```

- vLLM：

```bash
vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
```

本地使用方面，Ollama、LMStudio、MLX-LM、llama.cpp 和 KTransformers 等应用也已支持 Qwen3。

## 思考模式与非思考模式切换

> SGLang 和 vLLM 创建的 API 中也提供了 `enable_thinking` 开关。请参考我们为 [SGLang](https://qwen.readthedocs.io/en/latest/deployment/sglang.html#thinking-non-thinking-modes) 和 [vLLM](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes) 用户提供的文档。

### `enable_thinking=True`

默认情况下，Qwen3 启用了思考能力，类似于 QwQ-32B。这意味着模型会运用推理能力来提升生成回复的质量。例如，当显式设置 `enable_thinking=True` 或在 `tokenizer.apply_chat_template` 中保留默认值时，模型将进入思考模式。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # True 是 enable_thinking 的默认值
)
```

在此模式下，模型会生成包裹在 `<think>...</think>` 块中的思考内容，然后输出最终回复。

> 在思考模式下，使用 `Temperature=0.6`、`TopP=0.95`、`TopK=20`、`MinP=0`（`generation_config.json` 中的默认设置）。**不要使用贪心解码**，否则可能导致性能下降和无限重复。更详细的指导请参考[最佳实践](#最佳实践)部分。

### `enable_thinking=False`

我们提供了一个硬开关来严格禁用模型的思考行为，使其功能与之前的 Qwen2.5-Instruct 模型保持一致。此模式在需要禁用思考以提升效率的场景中特别有用。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 设置 enable_thinking=False 禁用思考模式
)
```

在此模式下，模型不会生成任何思考内容，也不会包含 `<think>...</think>` 块。

> 在非思考模式下，建议使用 `Temperature=0.7`、`TopP=0.8`、`TopK=20`、`MinP=0`。更详细的指导请参考[最佳实践](#最佳实践)部分。

### 高级用法：通过用户输入在思考与非思考模式间切换

我们提供了一个软切换机制，允许用户在 `enable_thinking=True` 时动态控制模型行为。具体来说，你可以在用户提示或系统消息中添加 `/think` 和 `/no_think` 来逐轮切换模型的思考模式。在多轮对话中，模型会遵循最近的指令。

以下是多轮对话的示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 更新历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# 使用示例
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # 第一次输入（不使用 /think 或 /no_think 标签，默认启用思考模式）
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # 第二次输入，使用 /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}")
    print("----------------------")

    # 第三次输入，使用 /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
```

> 为了 API 兼容性，当 `enable_thinking=True` 时，无论用户使用 `/think` 还是 `/no_think`，模型始终会输出一个 `<think>...</think>` 包裹块。但如果思考被禁用，该块的内容可能为空。当 `enable_thinking=False` 时，软切换无效。无论用户输入了任何 `/think` 或 `/no_think` 标签，模型都不会生成思考内容，也不会包含 `<think>...</think>` 块。

## Agent 使用

Qwen3 在工具调用能力方面表现出色。我们推荐使用 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) 来充分发挥 Qwen3 的 Agent 能力。Qwen-Agent 内部封装了工具调用模板和工具调用解析器，大大降低了编码复杂度。

定义可用工具时，你可以使用 MCP 配置文件、使用 Qwen-Agent 的内置工具，或自行集成其他工具。

```python
from qwen_agent.agents import Assistant

# 定义 LLM
llm_cfg = {
    'model': 'Qwen3-0.6B',

    # 使用阿里云百炼提供的端点：
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # 使用兼容 OpenAI API 的自定义端点：
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # 其他参数：
    # 'generate_cfg': {
    #         # 添加：当回复内容为 `<think>这是思考过程</think>这是答案` 时；
    #         # 不添加：当回复已经被分为 reasoning_content 和 content 时。
    #         'thought_in_content': True,
    #     },
}

# 定义工具
tools = [
    {'mcpServers': {  # 你可以指定 MCP 配置文件
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
    'code_interpreter',  # 内置工具
]

# 定义 Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# 流式生成
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ 介绍 Qwen 的最新进展'}]
for responses in bot.run(messages=messages):
    pass
print(responses)
```

## 最佳实践

为了达到最优性能，我们推荐以下设置：

1. **采样参数**：
   - 思考模式（`enable_thinking=True`）下，使用 `Temperature=0.6`、`TopP=0.95`、`TopK=20`、`MinP=0`。**不要使用贪心解码**，否则可能导致性能下降和无限重复。
   - 非思考模式（`enable_thinking=False`）下，建议使用 `Temperature=0.7`、`TopP=0.8`、`TopK=20`、`MinP=0`。
   - 对于支持的框架，可以将 `presence_penalty` 参数调整到 0 到 2 之间以减少无限重复。但使用较高的值可能偶尔导致语言混杂和模型性能轻微下降。

2. **充足的输出长度**：对于大多数查询，建议使用 32,768 token 的输出长度。对于高度复杂问题（如数学和编程竞赛题）的基准测试，建议将最大输出长度设置为 38,912 token。这为模型提供了足够的空间来生成详细和全面的回复，从而提升整体性能。

3. **标准化输出格式**：在基准测试时，建议使用提示词来标准化模型输出。
   - **数学题**：在提示词中加入 "Please reason step by step, and put your final answer within \\boxed{}."
   - **选择题**：在提示词中添加以下 JSON 结构来标准化回复："Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`."

4. **历史记录中不包含思考内容**：在多轮对话中，历史模型输出应仅包含最终输出部分，无需包含思考内容。这在提供的 Jinja2 聊天模板中已实现。但对于不直接使用 Jinja2 聊天模板的框架，需要开发者自行确保遵循此最佳实践。

## 引用

如果我们的工作对你有帮助，欢迎引用。

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report},
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388},
}
```

---

## 附加信息

- **上月下载量**：11,578,352
- **格式**：Safetensors
- **模型大小**：0.8B 参数
- **张量类型**：BF16
- **标签**：文本生成、Transformers、Safetensors、qwen3、对话、text-generation-inference
- **许可证**：Apache-2.0
- **论文**：arxiv:2505.09388
- **基座模型**：Qwen/Qwen3-0.6B-Base
- **模型衍生**：283 个适配器、691 个微调版本、10 个合并版本、259 个量化版本
- **合集**：Qwen3（84 项）
