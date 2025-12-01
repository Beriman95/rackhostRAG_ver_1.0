import subprocess
import json
from textwrap import dedent

def synthesize_answer(question: str, contexts: list[str]) -> str:
    prompt = dedent(f"""
    You are a Rackhost internal knowledgebase assistant.
    You MUST answer only from the provided context. Never add new information.

    QUESTION:
    {question}

    CONTEXT:
    {chr(10).join(contexts)}

    FORMAT:
    - Clear explanation
    - Steps if needed
    - Direct links if included in context
    """)

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()
