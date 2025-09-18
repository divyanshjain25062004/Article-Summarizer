Setup

Clone the repo:

git clone https://github.com/divyanshjain25062004/Article-Summarizer.git
cd Article-Summarizer


Create virtual environment:

python -m venv .venv
.venv\Scripts\activate  # on Windows
source .venv/bin/activate  # on macOS/Linux


Install dependencies:

pip install -r requirements.txt


Set your OpenAI API key:

setx OPENAI_API_KEY "your_api_key_here"   # Windows
export OPENAI_API_KEY="your_api_key_here" # macOS/Linux


Run the server:

uvicorn server:app --reload


Open in browser:

http://127.0.0.1:8000

Project Structure
Agentic-Strater/
│── agent.py          # Orchestrates search → fetch → summarize
│── tools.py          # Search, filtering, extraction, ranking
│── llm_client.py     # Handles LLM summarization calls
│── prompts.py        # Prompt templates for structured summaries
│── server.py         # FastAPI server (API + frontend)
│── static/           # Frontend (HTML/CSS/JS)
│── logs/             # Debug traces and error logs
│── requirements.txt  # Dependencies
│── README.md         # This file
