import ollama


OLLAMA_MODEL = "llama3.2"   # change here only


def run_ollama(prompt: str, temperature=0.2, max_tokens=256) -> str:
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_tokens
        }
    )
    return response["response"].strip()
