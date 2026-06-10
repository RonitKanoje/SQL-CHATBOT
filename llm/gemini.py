from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

GEMINI_LLM_MODEL = "gemini-2.5-flash-lite"

gemini_client = genai.Client()


def generate_text(prompt: str, temperature: float = 0.2) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig( #### what is this 
            temperature=temperature,
        ),
    )
    return response.text or ""


def generate_sql(prompt: str) -> str:
    return generate_text(prompt, temperature=0.0)


def explain_result(prompt: str) -> str:
    return generate_text(prompt, temperature=0.2)
