import os
import gradio as gr
import requests
import pandas as pd
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from bs4 import BeautifulSoup

# Scoring API endpoint
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# ====================== YOUR LLAMAINDEX AGENT ======================
class GaiaAgent:
    def __init__(self):
        print("Initializing LlamaIndex GaiaAgent via Serverless API...")
        
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI    

        # Pull the token from your local terminal session
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not found! Please set it in your terminal before running.")
        
        # Use serverless inference to query the model on HF's servers
        self.llm = HuggingFaceInferenceAPI(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            token=hf_token
        )
        
        Settings.llm = self.llm
        
        # Core Web Tools
        def web_search(query: str) -> str:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(query, max_results=5)]
                return str(results)
            except Exception as e:
                return f"Search error: {e}"
        
        def browse_page(url: str) -> str:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=" ")[:10000]
                return text
            except Exception as e:
                return f"Browse error: {e}"
        
        tools = [
            FunctionTool.from_defaults(fn=web_search, name="web_search", description="Search the internet for up-to-date information"),
            FunctionTool.from_defaults(fn=browse_page, name="browse_page", description="Fetch and read content from any URL"),
        ]
        
        self.agent = ReActAgent(
            tools=tools,
            llm=self.llm,
            verbose=True,
            max_iterations=15,
        )
        print("✅ LlamaIndex GaiaAgent ready!")

    def __call__(self, question: str) -> str:
        try:
            print(f"🤖 Processing: {question[:120]}...")
            response = self.agent.chat(question)
            final_answer = str(response).strip()
            
            # Formatting clean up to ensure answers match the strict exact-match grading
            if "Answer:" in final_answer:
                final_answer = final_answer.split("Answer:")[-1].strip()
            elif "final answer" in final_answer.lower():
                final_answer = final_answer.split("final answer", 1)[-1].strip(": \n")
            
            print(f"📤 Answer: {final_answer[:300]}...")
            return final_answer
        except Exception as e:
            print(f"❌ Error: {e}")
            return f"AGENT ERROR: {str(e)}"


# ====================== SUBMISSION LOGIC ======================
def run_and_submit_all(username: str, code_link: str):
    if not username.strip():
        return "Please enter your Hugging Face username to submit.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # Link to your code for verification (could be your GitHub repo, a gist, or a dummy URL)
    agent_code = code_link.strip() if code_link.strip() else "https://github.com/"

    # Fetch GAIA questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Error fetching questions: {e}", None

    # Run agent evaluation loop
    results_log = []
    answers_payload = []
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        return "No answers generated.", pd.DataFrame(results_log)

    # Submit payload back to the course endpoint
    submission_data = {
        "username": username.strip(), 
        "agent_code": agent_code, 
        "answers": answers_payload
    }
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = f"Submission Successful!\nScore: {result_data.get('score', 'N/A')}%"
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission failed: {e}", pd.DataFrame(results_log)


# Local Gradio Layout
with gr.Blocks() as demo:
    gr.Markdown("# Gaia Agent - Local Runner")
    gr.Markdown("This interface runs on your local machine and submits results directly to the course leaderboard.")
    
    username_input = gr.Textbox(label="Your Hugging Face Username", placeholder="e.g. yourusername")
    code_input = gr.Textbox(
        label="Code Link / Public Repository (Verification)", 
        placeholder="Link to your public repository with this code",
        value="https://github.com/"
    )
    
    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    status_output = gr.Textbox(label="Status", lines=6)
    results_table = gr.DataFrame(label="Results")

    run_button.click(
        fn=run_and_submit_all,
        inputs=[username_input, code_input],
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch()