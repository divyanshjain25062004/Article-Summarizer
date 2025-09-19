# Article Summarizer

## Getting started

Clone the repository and install dependencies inside a virtual environment:

```bash
git clone https://github.com/divyanshjain25062004/Article-Summarizer.git
cd Article-Summarizer
python -m venv .venv
source .venv/bin/activate            # ``.venv\\Scripts\\activate`` on Windows
pip install -r requirements.txt
```

Configure an OpenAI-compatible API key so the summarisation tool can talk to
the model:

```bash
export OPENAI_API_KEY="sk-..."         # use ``setx`` on Windows
```

Run the FastAPI server and open the interface:

```bash
uvicorn server:app --reload
# visit http://127.0.0.1:8000
```

## Agentic architecture

The project now ships as a lightweight agentic framework where tools are
first-class citizens and a planner chooses which capability to apply next.

* **`agent.py`** – Defines reusable agent primitives (`ToolSpec`,
  `AgentState`, and the generic `Agent` loop) plus the concrete
  `ArticleSummarizerAgent` wiring.
* **`tools.py`** – Implements search, extraction, and ranking primitives that
  are wrapped as agent tools.
* **`llm_client.py` / `prompts.py`** – Provide the LLM client and structured
  prompting used by the summarisation tool.
* **`server.py`** – FastAPI application exposing the `/search` endpoint and
  serving the frontend in `static/`.
* **`logs/`** – Populated at runtime with agent traces for debugging.

Because tools and planners are modular, you can register additional actions,
swap in a different planner (for instance an LLM-driven one), or orchestrate
multiple specialised agents while reusing the same building blocks.
