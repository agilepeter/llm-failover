"""Basic usage example for llm-failover.

Set your API keys as environment variables before running:
    export GROQ_API_KEY=your-key
    export GEMINI_API_KEY=your-key
"""

import logging
from llm_failover import LLMFailover
from llm_failover.providers import groq, gemini, cerebras, openrouter

# Enable logging to see failover in action
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- Option 1: Simple setup with built-in helpers ---

failover = LLMFailover([
    ("Groq", groq()),
    ("Gemini", gemini()),
])

result = failover.generate("What is the speed of light?")
print(result)


# --- Option 2: Production setup with budgets and blackout alerting ---

def on_blackout(chain, exhausted):
    print(f"ALL PROVIDERS DOWN! chain={chain}, exhausted={exhausted}")

failover2 = LLMFailover(
    providers=[
        ("Gemini", gemini()),
        ("Cerebras", cerebras()),
        ("Groq", groq()),
        ("OpenRouter", openrouter()),
    ],
    chains={
        "morning": ["Gemini", "Cerebras", "Groq", "OpenRouter"],
        "fast": ["Cerebras", "Gemini", "Groq", "OpenRouter"],
        "default": ["Groq", "Gemini", "Cerebras", "OpenRouter"],
    },
    budgets={
        "Groq": 8,
        "Gemini": 30,
        "Cerebras": 12,
        "OpenRouter": 50,
    },
    on_blackout=on_blackout,
)

result2 = failover2.generate("Summarize today's news", chain="morning")
print(result2)


# --- Option 3: Bring your own functions ---

def my_custom_provider(prompt, max_tokens=4096):
    """Any function with this signature works."""
    # Your API call here
    return "42"

failover3 = LLMFailover([
    ("Groq", groq()),
    ("Custom", my_custom_provider),
])
