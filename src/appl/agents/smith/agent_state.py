
from typing import Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    previous_response_id: Optional[str] = None


