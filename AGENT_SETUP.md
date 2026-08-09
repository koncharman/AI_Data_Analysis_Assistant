# Agent integration setup

Install the agent dependencies:

```bash
pip install -U -r requirements-agent.txt
```

Make sure Ollama is running and the selected model is installed:

```bash
ollama pull llama3.2:3b
ollama serve
```

Some local models are better at tool calling than others. If tool selection is
unreliable, choose another tool-capable Ollama model in the Streamlit sidebar.

Copy the `agents/` directory into the project root and replace `main.py` with
the supplied updated version.

Run:

```bash
streamlit run main.py
```

The full DataFrame remains in Python memory. The LLM receives only dataset
metadata and compact structured tool results.
