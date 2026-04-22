# llm-failover

Multi-provider LLM fallback with rate-limit handling and automatic retry. Zero vendor lock-in.

Born from running 4 AI agents daily across 5 LLM providers in production at [StaaS Fund](https://staas.fund). When one provider goes down or hits a rate limit, the next one picks up seamlessly.

## Features

- **Automatic failover** — tries providers in order, moves to the next on failure
- **Rate-limit detection** — catches 429 errors and retries with exponential backoff
- **Daily quota tracking** — when a provider hits its daily cap, it's skipped for the rest of the session
- **Round-robin rotation** — spreads calls across providers evenly instead of always hitting the first one
- **Per-provider budgets** — voluntarily rotate before hitting rate limits (e.g., limit Groq to 8 calls per session)
- **Named chains** — define "fast", "reliable", "morning" chains with different provider orders
- **Blackout callback** — hook into total failure events (e.g., alert Discord, PagerDuty)
- **Bring your own providers** — any `(prompt, max_tokens) -> str` function works
- **Built-in helpers** — pre-built functions for Groq, Gemini, SambaNova, Cloudflare Workers AI, OpenRouter
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
        ("SambaNova", sambanova()),
    ],
    chains={
        "fast": ["Groq", "Gemini"],             # speed priority
        "reliable": ["Gemini", "Groq"],         # quality priority
        "default": ["Groq", "Gemini", "SambaNova"],
    },
)

# Use a specific chain
result = failover.generate("Quick math: 2+2?", chain="fast")
```

## Per-Provider Budgets

Voluntarily rotate providers before hitting rate limits. After exhausting a provider's budget, the failover engine skips to the next one. Over-budget providers are retried as a last resort before total failure.

```python
failover = LLMFailover(
    providers=[
        ("Groq", groq()),
        ("Gemini", gemini()),
        ("SambaNova", sambanova()),
        ("OpenRouter", openrouter()),
    ],
    budgets={
        "Groq": 8,         # Conservative — daily token cap is the real constraint
        "Gemini": 30,       # Flash free tier has the most headroom
        "SambaNova": 6,     # Unreliable 500s — use sparingly
        "OpenRouter": 50,   # Paid final boss — only hit if everything else is down
    },
)
```

Calls are round-robin rotated across the chain automatically, so load spreads evenly even without budgets.

## Blackout Callback

Get notified when all providers fail:

```python
import requests

def alert_discord(chain, exhausted):
    requests.post(WEBHOOK_URL, json={
        "content": f"LLM blackout — all providers failed (chain={chain}, exhausted: {exhausted})"
    })

failover = LLMFailover(
    providers=[("Groq", groq()), ("Gemini", gemini())],
    on_blackout=alert_discord,
)
```

The callback fires once per session (not on every failed call). Call `failover.reset()` to re-arm it.

## Built-in Providers

| Provider | Helper | SDK | Env Var |
|----------|--------|-----|---------|
| Groq | `providers.groq()` | `pip install groq` | `GROQ_API_KEY` |
| Google Gemini | `providers.gemini()` | `pip install google-genai` | `GEMINI_API_KEY` |
| SambaNova | `providers.sambanova()` | None (REST) | `SAMBANOVA_API_KEY` |
| Cloudflare Workers AI | `providers.cloudflare()` | None (REST) | `CF_AI_API_TOKEN` + `CF_ACCOUNT_ID` |
| OpenRouter | `providers.openrouter()` | None (REST) | `OPENROUTER_API_KEY` |
| Any OpenAI-compatible | `providers.openai_compatible(base_url)` | None (REST) | `OPENAI_API_KEY` |

All helpers accept optional `api_key` and `model` overrides:

```python
from llm_failover.providers import groq

custom_groq = groq(api_key="sk-...", model="llama-3.1-8b-instant")
```

## How It Works

1. Round-robin rotate the chain so calls spread across providers
2. Try the next provider in the rotated chain (skip if over budget)
3. If it fails with a rate-limit error (429), retry with exponential backoff
4. If it hits a daily quota, mark it as exhausted and skip it for the session
5. If it fails for any other reason, move to the next provider
6. If all in-budget providers fail, retry over-budget providers as a last resort
7. If everything fails, fire the blackout callback and raise `RuntimeError`

Call `failover.reset()` to clear exhausted providers, call counts, and the blackout flag (e.g., at the start of a new day).

## Logging

Uses Python's standard `logging` module under the `llm_failover` logger:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## License

MIT
