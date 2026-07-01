from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool, StructuredTool
from sympy import latex
from sympy.parsing.latex import parse_latex
from .state import AgentState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import logging



log = logging.getLogger(__name__)

class ToolsWrapper:

    def __init__(self):
        pass

@tool("mathematical_calculations", parse_docstring=True)
def mathematical_calculations(expression: str) -> str:
    """Performs mathematical calculations.

    Args:
        expression (str): A string containing the mathematical expression to be evaluated, written in LaTeX format.

    Returns:
        str: The result of the calculation in LaTeX format.
    """
    log.debug(f"Evaluate expression:\n===================\n{expression}")
    try:
        expr1 = parse_latex(expression)
        log.debug(f"Parsed expression: {expr1}")

        eval_result = expr1.simplify()

        result = latex(eval_result)
        log.debug(f"========================Tool evaluation result: {result}")

        return result
    except Exception as e:
        log.error(f"Error solving equations: {e}")
        return str(e)


class NotebookTool:
    def __init__(self):
        self.content = []
        pass

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.notebook,
            name="notebook",
            description="Notebook tool for storing and retrieving any information",
            args_schema=NotebookArgs,
        )

    def notebook(self,
                 operation: str, 
                content: str,
                state: Annotated[AgentState, InjectedState],
                tool_call_id: Annotated[str, InjectedToolCallId],
                ) -> str:
        """Notebook tool for storing and retrieving any information.
        
        Args:
            operation (str): The operation to perform, either "store" or "retrieve".
            content (str): The content to store or retrieve (optional for "retrieve" operation).

        Returns:
            str: The stored content or retrieved notebook content.
        """
        
        log.debug(f"{'*'* 60 }\nNotebook operation: {operation}, content: {content}")
        normalized_operation = operation.strip().lower()
        store_ops = {"store", "save", "write", "append", "summarize", "add", "update", "insert", "record", "log", "note", "memo", "document", "register", "capture", "archive"}
        retrieve_ops = {"retrieve", "read", "get", "fetch", "load", "access", "view", "display", "show", "extract", "pull", "obtain", "collect", "recover", "uncover"}
        
        if normalized_operation in store_ops:
            log.debug(f"Storing content in notebook: {content}")
            
            self.content.append(content)
            
            return "OK"
        elif normalized_operation in retrieve_ops:
            content = state.notebook
            
            return content
        else:
            raise ValueError("Invalid operation. Use store/save/write/summarize or retrieve/read/get.")


class NotebookArgs(BaseModel):
    operation: Literal[
        "store",
        "read"
    ] = Field(description="Operation to perform")
    content: str = Field(
        default="",
        description="Text payload. Required for store-like operations and ignored for read-like operations",
    )
            