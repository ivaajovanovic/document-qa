import os
import json
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """Small helper around Gemini text generation API calls."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize Gemini client and validate required API key."""
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Generate a free-form text completion for a prompt."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


class GroqClient:
    """Helper for Groq chat completions with schema-constrained output."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        """Initialize Groq client and validate required API key."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel | None:
        """
        Request JSON output constrained by a schema and validate it with Pydantic.

        This method sends a strict JSON schema to the model and then validates
        the returned JSON payload by constructing the provided Pydantic model.

        Args:
            prompt: Input prompt.
            schema: Pydantic model class to validate against.

        Returns:
            Instance of the provided schema populated from model output.
        """
        # Build JSON Schema from Pydantic and disallow extra keys.
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
        # Parse JSON string returned by model and validate shape/types.
        data = json.loads(response.choices[0].message.content)
        return schema(**data)