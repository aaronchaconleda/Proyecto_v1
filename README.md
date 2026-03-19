# Proyecto_v1

RAG con dos perfiles en la misma estructura:
- Perfil `local` (LM Studio para embeddings y chat).
- Perfil `openai` (OpenAI para embeddings y chat).
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

## Configuracion `.env`

1. Crea `.env` en la raiz del proyecto (puedes copiar `.env.example`).
2. Define proveedor y credenciales:

```env
# Selector de perfil (recomendado)
RAG_PROFILE=local   # o openai

# Proveedor (opcional si ya usas RAG_PROFILE)
# local  -> lmstudio
# openai -> openai
RAG_LLM_PROVIDER=lmstudio

# Bases separadas por perfil (automatico por defecto):
# local  -> data/rag_local.db + data/chroma_local
# openai -> data/rag_openai.db + data/chroma_openai
# RAG_SQLITE_PATH=./data/rag_local.db
# RAG_CHROMA_DIR=./data/chroma_local

LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio

# Si usas OpenAI:
# RAG_LLM_PROVIDER=openai
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=tu_api_key
```

Notas:
- `.env` esta ignorado por git.
- Puedes mantener ambos bloques y cambiar solo `RAG_PROFILE` para alternar entre perfil local y OpenAI sin mezclar datos.

LM Studio debe estar corriendo con API OpenAI-compatible en:
- http://127.0.0.1:1234/v1

## Variables opcionales

- RAG_LLM_PROVIDER (default: lmstudio)
- RAG_PROFILE (default: local)
- RAG_SQLITE_PATH (default: data/rag_<profile>.db)
- RAG_CHROMA_DIR (default: data/chroma_<profile>)
- RAG_CHAT_MODEL (default segun perfil: `qwen/qwen3-4b-thinking-2507` en local, `gpt-5-nano` en openai)
- RAG_TEMPERATURE (default: 0.1)
- RAG_EMBEDDING_MODEL (default segun perfil: `text-embedding-nomic-embed-code` en local, `text-embedding-3-small` en openai)
- OPENAI_BASE_URL (default: https://api.openai.com/v1)
- OPENAI_API_KEY (requerida si RAG_LLM_PROVIDER=openai)
- RAG_CHUNK_SIZE (default: 500)
- RAG_CHUNK_OVERLAP_RATIO (default: 0.15)
- RAG_HYBRID (default: true)
- RAG_TOP_K (default: 6)
- RAG_VECTOR_WEIGHT (default: 0.7)
- RAG_KEYWORD_WEIGHT (default: 0.3)

## Uso rapido

### Interfaz chatbot (solo consulta)

La interfaz de chatbot esta orientada a usuario final para preguntar sobre documentos ya cargados en la RAG.
No incluye carga/borrado/indexado; esas operaciones siguen por CLI (admin).

Arranque:

```bash
chainlit run app/chatbot_ui.py -w
```

Flujo:
- Al iniciar, pide perfil `local` u `openai`.
- Conecta contra la base de ese perfil (`rag_local/chroma_local` o `rag_openai/chroma_openai`).
- Usa el pipeline de QA existente y responde con fuentes.

Requisito:
- Debe existir conocimiento ya indexado en el perfil elegido (por ejemplo con `python -m app.cli wizard`).

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

3. Simular borrado de documento obsoleto (verificas doc_id correcto, ver chunks/documento / No se modifica ni SQLite ni Chroma):

```bash
python -m app.cli delete-doc --doc-id Demo_SPA1
```

4. Borrado real de documento obsoleto:

```bash
python -m app.cli delete-doc --doc-id Demo_SPA1 --no-dry-run --confirm
```

5. Compactar SQLite principal (el archivo activo segun perfil, por ejemplo `rag_local.db` o `rag_openai.db`):

```bash
python -m app.cli vacuum-db
```

6. Compactar SQLite de Chroma (`chroma.sqlite3`):

```bash
python -m app.cli vacuum-chroma --confirm
```

Recomendacion: ejecutar `vacuum-chroma` sin chat/indexado activos para evitar bloqueos.

7. Indexar documento:

```bash
python -m app.cli index docs/Demo_SPA.pdf --doc-id demo_spa --lang es
```

Si no pasas `--embedding-model`, el comando abre selector interactivo de embedding.

Puedes forzar embedding model en indexado:

```bash
python -m app.cli index docs/Demo_SPA.pdf --doc-id demo_spa --embedding-model text-embedding-nomic-embed-code
```

8. Preguntar en chat:

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

## Robustez y trazabilidad

- Logs operativos en `data/logs/rag_cli.log`.
- Validacion de indice vacio antes de `chat`.
- Validacion de `doc_id_filter` contra documentos existentes.
- Validacion de modelos explicitos (`--chat-model`, `--embedding-model`) contra LM Studio.
- Mensajes de error claros para caida/no disponibilidad de LM Studio y bloqueos de SQLite.
