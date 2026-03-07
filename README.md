# llm-failover

Multi-provider LLM fallback with rate-limit handling and automatic retry. Zero vendor lock-in.

Born from running 4 AI agents daily across 5 LLM providers in production at [StaaS Fund](https://staas.fund). When one provider goes down or hits a rate limit, the next one picks up seamlessly.

## Features

- **Automatic failover** — tries providers in order, moves to the next on failure
- **Rate-limit detection** — catches 429 errors and retries with exponential backoff
- **Daily quota tracking** — when a provider hits its daily cap, it's skipped for the rest of the session
- **Named chains** — define "fast", "reliable", "morning" chains with different provider orders
- **Bring your own providers** — any `(prompt, max_tokens) -> str` function works
- **Built-in helpers** — pre-built functions for Groq, Gemini, Cerebras, SambaNova, Cloudflare Workers AI
- **Zero required dependencies** — core library is pure Python. Provider SDKs are optional.

## Install

```bash
pip install git+https://github.com/agilepeter/llm-failover.git
```

With built-in provider helpers:

```bash
pip install "git+https://github.com/agilepeter/llm-failover.git#egg=llm-failover[groq,gemini]"
```

## Quick Start

```python
from llm_failover import LLMFailover
from llm_failover.providers import groq, gemini

# Set GROQ_API_KEY and GEMINI_API_KEY as env vars, or pass directly
failover = LLMFailover([
    ("Groq", groq()),
    ("Gemini", gemini()),
])

result = failover.generate("Explain quantum computing in one sentence")
print(result)
```

## Bring Your Own Provider

Any function with the signature `(prompt: str, max_tokens: int) -> str` works:

```python
import openai

def my_openai(prompt, max_tokens=4096):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

failover = LLMFailover([
    ("GPT-4o-mini", my_openai),
    ("Groq", groq()),
])
```

## Named Chains

Define multiple chains for different scenarios:

```python
failover = LLMFailover(
    providers=[
        ("Groq", groq()),
        ("Gemini", gemini()),
        ("Cerebras", cerebras()),
    ],
    chains={
        "fast": ["Cerebras", "Groq"],       # speed priority
        "reliable": ["Gemini", "Groq"],       # quality priority
        "default": ["Groq", "Gemini", "Cerebras"],
    },
)

# Use a specific chain
result = failover.generate("Quick math: 2+2?", chain="fast")
```

## Built-in Providers

| Provider | Helper | SDK | Env Var |
|----------|--------|-----|---------|
| Groq | `providers.groq()` | `pip install groq` | `GROQ_API_KEY` |
| Google Gemini | `providers.gemini()` | `pip install google-genai` | `GEMINI_API_KEY` |
| Cerebras | `providers.cerebras()` | `pip install cerebras-cloud-sdk` | `CEREBRAS_API_KEY` |
| SambaNova | `providers.sambanova()` | None (REST) | `SAMBANOVA_API_KEY` |
| Cloudflare Workers AI | `providers.cloudflare()` | None (REST) | `CF_AI_API_TOKEN` + `CF_ACCOUNT_ID` |
| Any OpenAI-compatible | `providers.openai_compatible(base_url)` | None (REST) | `OPENAI_API_KEY` |

All helpers accept optional `api_key` and `model` overrides:

```python
from llm_failover.providers import groq

custom_groq = groq(api_key="sk-...", model="llama-3.1-8b-instant")
```

## How It Works

1. Try the first provider in the chain
2. If it fails with a rate-limit error (429), retry with exponential backoff
3. If it hits a daily quota, mark it as exhausted and skip it for the session
4. If it fails for any other reason, move to the next provider
5. If all providers fail, raise `RuntimeError`

Call `failover.reset()` to clear exhausted providers (e.g., at the start of a new day).

## Logging

Uses Python's standard `logging` module under the `llm_failover` logger:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## License

MIT
