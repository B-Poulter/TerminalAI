# Terminal AI Connectors

A lightweight, terminal-native CLI tool for advanced AI chat interfaces—powered by **Google Gemini** or **local Ollama models**. Designed for Linux (optimized for Debian / KDE Plasma), it offers live Markdown rendering, visual thinking-process animations, file attachments, and security-focused system tool execution.

> ⚠️ **IMPORTANT SECURITY NOTICE:** These connectors grant the AI access to a non-root shell on your system. The AI can execute commands, read files, and interact with your OS. Always review commands carefully before execution.

---

## ✨ Key Features

- **Terminal-Native Workflow:** Stream responses with full Markdown rendering—including headings, tables, syntax-highlighted code blocks, and formatted lists—directly in your terminal.
- **Dual-Backend Support:**
  - `gemini_cli.py`: Connects to Google's hosted models (default: `gemini-3.8-flash`).
  - `ollama_cli.py`: Connects to local models via Ollama with automatic capabilities detection (Vision, Tool Calling, and Reasoning/Thinking tokens).
- **Controlled Shell Execution:** The AI can generate and execute Bash commands within a restricted (non-root) environment to automate system tasks.
- **Interactive Command Approval:** Prioritizes security by pausing execution to request explicit user approval before running any terminal command.
- **Built-in Tools:** Integrated capabilities for web search (Google Search grounding, DuckDuckGo, or SearXNG), web page browsing, local timers, and reminders.
- **File Attachments:** Easily attach files using `@file` shortcuts, the `/attach` command, or by dragging and dropping files directly into the terminal window.

---

## 🛡️ Security Guidelines & Best Practices

Because these scripts grant the AI access to a local shell, ambiguous prompts or incorrect AI outputs could lead to data modification or loss. Follow these guidelines for safe operation:

- **Review Every Command:** The tool prompts for confirmation prior to executing any command. Only approve commands you thoroughly understand. Reject anything suspicious or unclear.
- **Never Run as Root:** Do not launch these connectors using `sudo` or under a `root` account. Execution as root is automatically blocked by the script.
- **Isolate Your Environment:** Avoid running these connectors on critical production systems or machines containing sensitive personal data.
- **Use Sandboxing:** For optimal security, run the scripts inside a dedicated **Docker container** or an isolated Virtual Machine (VM).

---

## 📋 Prerequisites

- **Python 3.11+**
- **Optional Dependencies:** `pygments` (`pip install pygments`) for rich code syntax highlighting.
- **Backend Requirements:**
  - **For Gemini:** An API key from [Google AI Studio](https://aistudio.google.com/apikey).
  - **For Ollama:** A running local Ollama instance (`ollama serve`) with your preferred models pulled.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/terminal-ai-connectors.git](https://github.com/yourusername/terminal-ai-connectors.git)
   cd terminal-ai-connectors
