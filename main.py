from langchain.agents import initialize_agent, load_tools
from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model_name=os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-07-01-preview",
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

tools = load_tools(["serpapi"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def run_agent(query: str):
    return agent.run(query)

if __name__ == "__main__":
    result = run_agent("Tell me a fun fact about space.")
    print(result)
