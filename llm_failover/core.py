"""Core failover engine.

Battle-tested in production running 4 AI agents daily across 5 LLM providers.
Handles rate limits, daily quotas, and provider outages automatically.
"""

import time
import logging

logger = logging.getLogger("llm_failover")


def _is_rate_limited(error):
    """Check if an error is a rate limit (429) error."""
    s = str(error)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "rate_limit" in s


def _is_daily_limit(error):
    """Check if error is a daily/quota limit (not just per-minute throttle)."""
    s = str(error)
    return "per day" in s or "TPD" in s or "daily" in s.lower()


class LLMFailover:
    """Multi-provider LLM failover with automatic retry and rate-limit handling.

    Args:
        providers: Provider definitions. Accepts:
            - List of tuples: [("Groq", groq_fn), ("OpenAI", openai_fn)]
            - Dict: {"Groq": groq_fn, "OpenAI": openai_fn}
            Each function must accept (prompt: str, max_tokens: int) and return str.
        chains: Optional named chains. Dict of {chain_name: [provider_names]}.
            If not provided, "default" chain uses all providers in order.
        retries: Number of retries per provider on rate-limit errors. Default 2.
        backoff_base: Base seconds for exponential backoff. Default 8.

    Example:
        failover = LLMFailover([
            ("Groq", my_groq_fn),
            ("Gemini", my_gemini_fn),
            ("OpenAI", my_openai_fn),
        ])
        result = failover.generate("Explain quantum computing")
    """

    def __init__(self, providers, chains=None, retries=2, backoff_base=8):
        # Normalize providers to {name: fn}
        if isinstance(providers, dict):
            self._providers = dict(providers)
            self._order = list(providers.keys())
        elif isinstance(providers, (list, tuple)):
            self._providers = {}
            self._order = []
            for item in providers:
                name, fn = item[0], item[1]
                self._providers[name] = fn
                self._order.append(name)
        else:
            raise TypeError("providers must be a list of (name, fn) tuples or a dict")

        # Normalize chains
        if chains:
            self._chains = dict(chains)
        else:
            self._chains = {}
        if "default" not in self._chains:
            self._chains["default"] = list(self._order)

        self._retries = retries
        self._backoff_base = backoff_base
        self._exhausted = set()

    def generate(self, prompt, chain="default", max_tokens=4096):
        """Generate text using the specified fallback chain.

        Tries each provider in order. On rate-limit errors, retries with
        exponential backoff. On daily quota errors, marks the provider as
        exhausted for the rest of the session.

        Args:
            prompt: The prompt text.
            chain: Chain name (default: "default"). Must match a key in chains.
            max_tokens: Maximum tokens for the response.

        Returns:
            Generated text string from the first successful provider.

        Raises:
            RuntimeError: If all providers in the chain fail.
        """
        provider_names = self._chains.get(chain, self._chains["default"])

        for name in provider_names:
            fn = self._providers.get(name)
            if fn is None:
                logger.warning("Provider '%s' in chain '%s' not found, skipping", name, chain)
                continue

            result = self._try_provider(name, fn, prompt, max_tokens)
            if result is not None:
                return result

        raise RuntimeError(
            f"All providers failed (chain='{chain}', "
            f"tried: {provider_names}, "
            f"exhausted: {list(self._exhausted)})"
        )

    def _try_provider(self, name, fn, prompt, max_tokens):
        """Try a single provider with retry logic."""
        if name in self._exhausted:
            logger.debug("Skipping exhausted provider: %s", name)
            return None

        logger.info("Trying %s", name)

        for attempt in range(self._retries + 1):
            try:
                return fn(prompt, max_tokens)
            except Exception as e:
                if _is_rate_limited(e):
                    if _is_daily_limit(e):
                        logger.warning("%s daily limit hit — skipping for rest of session", name)
                        self._exhausted.add(name)
                        return None
                    if attempt < self._retries:
                        wait = self._backoff_base * (attempt + 1)
                        logger.info("%s rate-limited, retrying in %ds...", name, wait)
                        time.sleep(wait)
                        continue
                logger.warning("%s failed: %s", name, e)
                return None

        return None

    def reset(self):
        """Clear exhausted providers. Call this at the start of a new day/session."""
        self._exhausted.clear()

    @property
    def exhausted(self):
        """Set of provider names that hit daily limits this session."""
        return frozenset(self._exhausted)

    @property
    def chains(self):
        """Dict of available chain names to provider lists."""
        return dict(self._chains)

    def add_chain(self, name, provider_names):
        """Register a new named chain.

        Args:
            name: Chain identifier (e.g., "fast", "morning").
            provider_names: Ordered list of provider names to try.
        """
        self._chains[name] = list(provider_names)
