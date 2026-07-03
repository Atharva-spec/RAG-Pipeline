import ollama

from config.settings import LLM_MODEL, LLM_TEMPERATURE


def get_answer(messages: list[dict]) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": LLM_TEMPERATURE},
    )
    return response["message"]["content"]
