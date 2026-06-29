from appl.domain import ContextType, MessagesBag, MessageContextType, ContextMessage, MessageContentType
from enum import Enum
from typing import Dict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from appl.domain import StudentProfile


# The category of the solution approach.
class SolutionCategory(str, Enum):
    UNKNOWN = "UNKNOWN"
    CALC = "CALC" # solve and give the answer directly
    CALC_CHOOSE = "CALC_CHOOSE" # solve and choose from multiple choices


class MathProblemDefinition(BaseModel):
    problem: str = Field(..., description="The mathematical problem to be solved, represented as a string.")

    solution_category: SolutionCategory = Field(default=SolutionCategory.CALC, description="The category of the solution approach.")


class MathProblemState(BaseModel):
    messages: Annotated[list, add_messages]  = Field(default_factory=list)

    problem: MathProblemDefinition = Field(..., description="The mathematical problem to be solved.")

    student_profile: Optional[StudentProfile] = Field(default=None)

    previous_response_id: Optional[str] = Field(default = None)
