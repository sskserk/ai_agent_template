import logging
from langchain_openai import ChatOpenAI
from agent.agent_smith import AgentSmith
from agent.tools import NotebookTool
from pathlib import Path

log = logging.getLogger(__name__)

def setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file =  "app.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    log.info("Logging initialized. Output file: %s", log_file)


chat_model = ChatOpenAI(
            model="gpt-4.1-mini", 
#            reasoning_effort="medium",
        )



setup_logging()

notebook_tool = NotebookTool()
    
math_agent = AgentSmith(notebook_tool=notebook_tool, chat_model=chat_model)
res = math_agent.invoke("""Calculate the 2+2*3.""")

print("Final response:==============")
for m in res.messages:
    print(f"Message. Type: {m.type}, Content: {m.content}")
    
print("Notebook content:==============:\n", "\n".join(notebook_tool.content))