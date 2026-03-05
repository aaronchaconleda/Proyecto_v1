# Proyecto_v1

RAG local con:
- LM Studio para embeddings y chat.
- ChromaDB para busqueda vectorial.
- SQLite para documentos, chunks, sesiones e historial.

## Estructura

- src/config: settings y prompts.
- src/ingestion: carga de PDF/TXT/MD y limpieza.
- src/chunking: chunking semantico y por ventana.
- src/storage: SQLite, Chroma y sincronizacion.
- src/retrieval: vector, keyword (FTS) e hibrido.
- src/llm: cliente LM Studio y generacion.
- src/pipeline: indexacion y QA.
- src/eval: utilidades de evaluacion.
- app/cli.py: interfaz CLI.

## Requisitos

```bash
pip install -r requirements.txt
```

LM Studio debe estar corriendo con API OpenAI-compatible en:
- http://127.0.0.1:1234/v1

## Variables opcionales

- RAG_CHAT_MODEL (default: qwen/qwen3-4b-thinking-2507)
- RAG_EMBEDDING_MODEL (default: text-embedding-nomic-embed-code)
- RAG_CHUNK_SIZE (default: 500)
- RAG_CHUNK_OVERLAP_RATIO (default: 0.15)
- RAG_HYBRID (default: true)
- RAG_TOP_K (default: 6)
- RAG_VECTOR_WEIGHT (default: 0.7)
- RAG_KEYWORD_WEIGHT (default: 0.3)

## Uso rapido

### Opcion A: flujo guiado (un solo comando)

```bash
python -m app.cli wizard
```

El asistente de consola te pedira:
- Modelo conversacional (listado desde LM Studio).
- Modelo de embedding (listado desde LM Studio).
- Ruta del documento.
- `session_id`, `doc_id`, idioma y `top_k`.

Luego indexa y abre modo chat en la misma consola.

Si ya tienes conocimiento cargado y no quieres subir documento nuevo:

```bash
python -m app.cli wizard --no-index
```

Durante `wizard` puedes indicar filtro opcional de `doc_id` (uno o varios separados por coma).

### Opcion B: comandos separados

1. Crear sesion:

```bash
python -m app.cli init-session --session-id demo-es
```

2. Ver documentos indexados:

```bash
python -m app.cli list-docs
```

3. Indexar documento:

```bash
python -m app.cli index docs/Demo_SPA.pdf --doc-id demo_spa --lang es
```

Si no pasas `--embedding-model`, el comando abre selector interactivo de embedding.

Puedes forzar embedding model en indexado:

```bash
python -m app.cli index docs/Demo_SPA.pdf --doc-id demo_spa --embedding-model text-embedding-nomic-embed-code
```

3. Preguntar en chat:

```bash
python -m app.cli chat --session-id demo-es "Cual es el objetivo principal del documento?"
```

Si no pasas `--chat-model`, el comando abre selector interactivo de LLM.

Filtrar consulta por documento especifico:

```bash
python -m app.cli chat --session-id demo-es --doc-id-filter Demo_SPA1 "Que dice sobre el administrador?"
```

Filtrar por varios documentos:

```bash
python -m app.cli chat --session-id demo-es --doc-id-filter Demo_SPA1,Demo_SPA2 "Compara los requisitos."
```

Cambiar modelo conversacional por consulta:

```bash
python -m app.cli chat --session-id demo-es --chat-model mistralai/mistral-3-3b "Resume la seccion de alcance."
```
