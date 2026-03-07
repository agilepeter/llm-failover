"""Basic usage example for llm-failover.

Set your API keys as environment variables before running:
    export GROQ_API_KEY=your-key
    export GEMINI_API_KEY=your-key
"""

import logging
from llm_failover import LLMFailover
from llm_failover.providers import groq, gemini

# Enable logging to see failover in action
logging.basicConfig(level=logging.INFO, format="%(message)s")

# --- Option 1: Use built-in provider helpers ---

failover = LLMFailover([
    ("Groq", groq()),
    ("Gemini", gemini()),
])

result = failover.generate("What is the speed of light?")
print(result)


# --- Option 2: Bring your own functions ---

def my_custom_provider(prompt, max_tokens=4096):
    """Any function with this signature works."""
    # Your API call here
    return "42"

failover2 = LLMFailover([
    ("Groq", groq()),
    ("Custom", my_custom_provider),
])


# --- Option 3: Named chains for different use cases ---

failover3 = LLMFailover(
    providers=[
        ("Groq", groq()),
        ("Gemini", gemini()),
    ],
    chains={
        "fast": ["Groq"],              # speed priority
        "reliable": ["Gemini", "Groq"],  # quality priority
    },
)

fast_result = failover3.generate("Quick answer: 2+2?", chain="fast")
print(fast_result)
