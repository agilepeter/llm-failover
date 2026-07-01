"""Built-in provider helpers for common LLM services.

Each function returns a callable with signature (prompt, max_tokens) -> str,
ready to pass directly to LLMFailover.

All providers use lazy imports — SDKs are only imported when the function is
first called, so you won't get import errors for providers you don't use.
"""

import os


def groq(api_key=None, model="llama-3.3-70b-versatile", timeout=120):
    """Create a Groq provider function.

    Requires: pip install groq
    """
    key = api_key or os.getenv("GROQ_API_KEY")

    def _generate(prompt, max_tokens=4096):
        from groq import Groq
        client = Groq(api_key=key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    return _generate


def anthropic(api_key=None, model="claude-haiku-4-5", timeout=120):
    """Create an Anthropic Claude provider function.

    Requires: pip install anthropic
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def _generate(prompt, max_tokens=4096):
        from anthropic import Anthropic
        client = Anthropic(api_key=key, timeout=timeout)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # content is a list of blocks; return the first text block
        return next((b.text for b in response.content if b.type == "text"), "")

    return _generate


def gemini(api_key=None, model="gemini-2.5-flash", timeout=120_000):
    """Create a Google Gemini provider function.

    Requires: pip install google-genai
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    _client = [None]  # mutable container for lazy init

    def _generate(prompt, max_tokens=4096):
        if _client[0] is None:
            from google import genai
            _client[0] = genai.Client(api_key=key)
        response = _client[0].models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"http_options": {"timeout": timeout}},
        )
        return response.text

    return _generate


def sambanova(api_key=None, model="Meta-Llama-3.3-70B-Instruct", timeout=120):
    """Create a SambaNova provider function.

    Uses the OpenAI-compatible REST API. No extra SDK needed.
    """
    import requests
    key = api_key or os.getenv("SAMBANOVA_API_KEY")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate


def cloudflare(api_token=None, account_id=None,
               model="@cf/meta/llama-3.3-70b-instruct-fp8-fast", timeout=120):
    """Create a Cloudflare Workers AI provider function.

    Uses the Cloudflare AI REST API. No extra SDK needed.
    """
    import requests
    token = api_token or os.getenv("CF_AI_API_TOKEN")
    acct = account_id or os.getenv("CF_ACCOUNT_ID")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate


def openrouter(api_key=None, model="meta-llama/llama-3.3-70b-instruct", timeout=120):
    """Create an OpenRouter provider function.

    Uses the OpenRouter REST API. No extra SDK needed.
    Free and paid models available at https://openrouter.ai/models
    """
    import requests
    key = api_key or os.getenv("OPENROUTER_API_KEY")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate


def github_models(api_key=None, model="openai/gpt-4o-mini", timeout=120):
    """Create a GitHub Models provider function.

    Uses the OpenAI-compatible REST API. No extra SDK needed.
    Free tier — authenticate with a GitHub token that has the `models:read`
    scope (e.g. from `gh auth token` or a PAT).
    Model catalog at https://github.com/marketplace/models
    """
    import requests
    key = api_key or os.getenv("GITHUB_TOKEN")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            "https://models.github.ai/inference/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate


def nvidia(api_key=None, model="meta/llama-3.1-8b-instruct", timeout=120):
    """Create an NVIDIA NIM provider function.

    Uses the OpenAI-compatible REST API. No extra SDK needed.
    Free tier + model catalog at https://build.nvidia.com
    Defaults to the 8B model — it stays warm and responds in <1s. Larger
    models (e.g. meta/llama-3.3-70b-instruct) can cold-start past the timeout.
    """
    import requests
    key = api_key or os.getenv("NVIDIA_API_KEY")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate


def mistral(api_key=None, model="mistral-small-latest", timeout=120):
    """Create a Mistral (La Plateforme) provider function.

    Uses the OpenAI-compatible REST API. No extra SDK needed.
    Free experimental tier available at https://console.mistral.ai
    """
    import requests
    key = api_key or os.getenv("MISTRAL_API_KEY")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate


def openai_compatible(base_url, api_key=None, model="gpt-4o-mini", timeout=120):
    """Create a provider for any OpenAI-compatible API (OpenAI, Together, Anyscale, etc.).

    Uses raw HTTP requests. No SDK needed.
    """
    import requests
    key = api_key or os.getenv("OPENAI_API_KEY")

    def _generate(prompt, max_tokens=4096):
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _generate
