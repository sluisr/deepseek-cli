# DeepSeek CLI

> Unofficial fork of Google Gemini CLI adapted to use the [DeepSeek API](https://platform.deepseek.com/). Original work Copyright 2025 Google LLC — Adaptations Copyright 2026 sluisr — Apache 2.0 License.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![NPM Version](https://img.shields.io/npm/v/@sluisr/deepseek-cli)](https://www.npmjs.com/package/@sluisr/deepseek-cli)

DeepSeek CLI is an open-source AI agent that brings the power of DeepSeek directly into your terminal. High-performance autonomous coding, tool execution, and context management by [sluisr](https://sluisr.com/).

---

## Why DeepSeek CLI?

- **Pay-per-use & Cost-Efficient:** Pay only for tokens used via DeepSeek API with up to 90% KV Cache discount.
- **DeepSeek V4 Models:** Full support for `deepseek-v4-flash` and `deepseek-v4-pro` (Reasoning / Thinking CoT mode).
- **Fast Code Patching:** Atomic unified diff patching (`apply_patch`) for instant, token-efficient code modifications.
- **Built-in Developer Tools:** File system operations, native server-side web search, shell execution, memory persistence, and FIM autocompletion.
- **Silent Sudo & System Auth:** Native non-blocking zero-lag AskPass integration for `sudo`, `ssh`, and `git`.
- **MCP Extensible:** Full Model Context Protocol (MCP) server support.
- **Isolated Configuration:** Operates under `~/.deepseek/` without polluting system or official CLI configurations.

---

## Installation

### Quick Install (NPM Global)

```bash
npm install -g @sluisr/deepseek-cli
```

Then run:

```bash
deepseek
```

### Update to Latest Version

```bash
npm install -g @sluisr/deepseek-cli@latest
```

---

## Authentication

Get your API key at [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) and export it:

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
deepseek
```

Or enter it interactively when prompted on first launch.

Optional custom base URL or proxy:

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

---

## Getting Started

### Interactive Session

```bash
cd my-project/
deepseek
```

### Non-Interactive (Headless Scripting)

```bash
deepseek -p "Explain the architecture of this codebase"
```

### Include Multiple Directories

```bash
deepseek --include-directories ../lib,../docs
```

---

## Slash Commands Reference

| Command | Description |
| :--- | :--- |
| `/model` | Interactive model selector (`deepseek-v4-flash`, `deepseek-v4-pro`) with real-time reasoning effort (`LOW`, `MEDIUM`, `HIGH`, `MAX`) and temperature controls. |
| `/balance` | Live DeepSeek account balance lookup (`GET /user/balance`). Alias: `/wallet`, `/credits`. |
| `/fim <file> [<line> \| <FIM_HOLE>]` | Fill-in-the-Middle code autocompletion via DeepSeek Beta API. |
| `/prefix <text>` | Chat Prefix Completion to force exact output format without conversational filler. |
| `/info` | Author credits, version info, and community links. Alias: `/author`, `/credits`. |
| `/chat` / `/resume` | Search and resume previous conversation sessions. |
| `/rewind` | Rewind conversation history to an earlier turn. |
| `/compress` | Compress conversation context to stay within token limits. |
| `/mcp` | Manage Model Context Protocol (MCP) servers and tools. |
| `/skills` | Manage and activate agent skills. |
| `/plan` | Enter interactive planning mode for complex multi-file architectural changes. |
| `/stats` | View session token usage, cached token counts, and execution metrics. |
| `/clear` | Clear terminal conversation screen. |
| `/quit` / `/exit` | Exit the CLI session and generate resume command (`deepseek --resume <id>`). |

---

## Built-in Agent Tools

DeepSeek CLI equips the AI agent with native developer tools to autonomously analyze, edit, and build projects:

- **Atomic Code Patching (`apply_patch`):** Fast, unified diff patching saving up to 80% tokens compared to full file rewrites.
- **Server-Side Web Search (`web_search`):** DeepSeek Responses API search engine with direct citation links and multi-level reasoning verification (`search_reasoning: low | medium | high | max`).
- **File System Operations:** Read, write, smart-replace, search with ripgrep (`grep`), and directory discovery (`glob`).
- **Shell Command Execution:** Autonomous bash commands, test runners, and builds with silent 0ms AskPass for `sudo`, `ssh`, and `git`.
- **Memory & Context Files:** Persistent agent memory in `~/.deepseek/DEEPSEEK.md` and per-project `DEEPSEEK.md`.
- **MCP Server Protocol:** Integrations with databases, GitHub, Docker, and custom tools via Model Context Protocol.

---

## Configuration

Configuration and persistence settings are stored under `~/.deepseek/settings.json`.

### MCP Servers Configuration Example

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

---

## Community & Author

- **Developer:** [sluisr](https://sluisr.com/)
- **GitHub:** [https://github.com/sluisr](https://github.com/sluisr)
- **Repository:** [https://github.com/sluisr/deepseek-cli](https://github.com/sluisr/deepseek-cli)
- **YouTube:** [https://www.youtube.com/@sluisr_](https://www.youtube.com/@sluisr_)

---

## Contributing

Issues and pull requests are welcome at [github.com/sluisr/deepseek-cli](https://github.com/sluisr/deepseek-cli).

---

## Legal & License

- **License:** [Apache License 2.0](LICENSE)
- **Original Project:** [Google Gemini CLI](https://github.com/google-gemini/gemini-cli) (Copyright 2025 Google LLC)
- **Adaptations & Port:** Copyright 2026 sluisr

---

<p align="center">
  Built by <a href="https://sluisr.com">sluisr</a> (<a href="https://github.com/sluisr">@sluisr</a>) — Unofficial DeepSeek CLI
</p>
