import os

import openai


class LLMClient:
    """Simple OpenAI-based LLM client for prompt generation."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set to use LLMClient.")

        openai.api_key = self.api_key
        self.model = model or os.getenv("GROQ_MODEL", "gpt-3.5-turbo")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
