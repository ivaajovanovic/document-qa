import os
import json
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """Wrapper around Gemini API for text completion."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


class GroqClient:
    """Wrapper around Groq API for text completion."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel | None:
        """
        Send prompt to Groq and return structured output validated against Pydantic schema.
        Uses strict JSON schema enforcement — server guarantees response matches the schema.

        Args:
            prompt: Input prompt.
            schema: Pydantic model class to validate against.

        Returns:
            Pydantic model instance if successful, None if failed.
        """
        schema_dict = schema.model_json_schema()
        schema_dict["additionalProperties"] = False

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema_dict
                }
            }
        )
        data = json.loads(response.choices[0].message.content)
        return schema(**data)