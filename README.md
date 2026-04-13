# WeChat LLM Proxy

一个开源的 WeChat→LLM 反向代理。让你在 WeChat 中直接与 Claude、GPT 等大语言模型交互。

## 快速开始

### 环境要求

- Python 3.11+
- uv：`curl -LsSf https://astral.sh/uv/install.sh | sh`

### 部署

```bash
git clone https://github.com/ha-ji-mu/wechat-llm-proxy.git
cd wechat-llm-proxy
uv sync
cp config/config.example.yaml config/config.yaml
nano config/config.yaml  # 找到 api_key: "${ANTHROPIC_API_KEY}" 这一行，改成 api_key: "sk-ant-xxxxx"
uv run python -m src.main --config config/config.yaml
```

首次运行会生成二维码，用 WeChat 扫描完成登录。

## 使用

在 WeChat 中向 Bot 发送：

```
!!              开始新对话
!!!             查看模型和 token 用量
!model:sonnet   切换模型
!help           显示帮助
```

## 许可证

MIT
