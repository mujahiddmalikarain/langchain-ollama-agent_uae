from fastapi import FastAPI
from pydantic import BaseModel, Field
from agent.main_agent import create_main_agent
import uvicorn

app = FastAPI(
    title="LangChain Agent API",
    description="Ask about weather or real estate developers in Dubai using a local LLM.",
    version="1.0.0"
)

agent = create_main_agent()

class PromptInput(BaseModel):
    prompt: str = Field(..., example="Tell me about AB Developers in Dubai Marina")

@app.post("/query")
def query_agent(input: PromptInput):
    result = agent.run(input.prompt)
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
