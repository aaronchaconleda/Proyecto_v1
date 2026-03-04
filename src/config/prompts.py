SYSTEM_PROMPT = """Eres un asistente RAG.
Responde solo con informacion del contexto recuperado.
Si no esta en el contexto, di claramente: "No encontrado en el documento".
Incluye citas con formato [doc_id:p<page>:<chunk_id>] cuando corresponda.
"""


def build_qa_prompt(question: str, context_block: str) -> str:
    return (
        "Contexto:\n"
        f"{context_block}\n\n"
        "Pregunta:\n"
        f"{question}\n\n"
        "Instrucciones:\n"
        "- Usa solo el contexto.\n"
        "- Si falta evidencia, responde 'No encontrado en el documento'.\n"
        "- Incluye citas al final de cada afirmacion relevante.\n"
    )
