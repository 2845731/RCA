from __future__ import annotations

from typing import Any, Dict, List, Optional


class LLMClient:
    """Unified LLM API client supporting OpenAI-compatible interfaces.

    Supports:
    - OpenAI API (gpt-4, gpt-3.5-turbo, etc.)
    - Any OpenAI-compatible API (e.g., Azure OpenAI, local vLLM, Ollama)
    - Fallback to rule-based mode when no API key is configured
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._available = False

        if self.api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
                self._available = True
            except ImportError:
                print("[LLMClient] openai package not installed. Run: pip install openai")
                self._available = False
            except Exception as e:
                print(f"[LLMClient] Failed to initialize: {e}")
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return the assistant's response text.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text, or empty string if unavailable.
        """
        if not self._available:
            return ""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLMClient] API call failed: {e}")
            return ""

    def chat_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Convenience method: send system + user messages.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user query.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
