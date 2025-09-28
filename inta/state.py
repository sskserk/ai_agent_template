
from typing import Dict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]  = Field(default_factory=list)


