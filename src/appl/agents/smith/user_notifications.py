
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


NotificationType = Literal[
    "streamed_chunk",
    "tool_start",
    "tool_end",
    "final_response",
    "user",
    "function_call",
]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    message_id: Optional[str] = Field(default=None)

class UserNotification(BaseModel):
    message_id: Optional[str] = Field(default=None)
    message: Any
    type: NotificationType
    end: Optional[str] = Field(default="\n")
    
    is_start: Optional[bool] = Field(default=False)
    is_end: Optional[bool] = Field(default=False)

