import logging
import datetime
import asyncio
import os
from typing import Awaitable, Callable, Literal, Optional

from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from .states import AgentStateWrapper
from .agent_state import AgentState
from .user_notifications import ChatMessage, UserNotification
from .tools import mathematical_calculations, expressions_logger

log = logging.getLogger(__name__)


POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
)


model_code = "gpt-5-mini-2025-08-07"

model_code = "gpt-4.1"

chat_model = ChatOpenAI(
    model=model_code,
    use_responses_api=True,
    use_previous_response_id=True,
    output_version="responses/v1",
#    reasoning={"effort": "medium"},
)


class AgentSmith:
    def __init__(self, 
                 thread_id: str,
                 message_printer: Optional[Callable[[UserNotification], Awaitable[None]]] = None,
                 ):
        if message_printer is None:
            raise ValueError("message_printer must be provided and cannot be None.")
    
        self._message_printer = message_printer
        self._thread_id = f"math_agent_{thread_id}"
        self.checkpointer = None
        self._checkpointer_conn = None
        self._checkpointer_ready = False
        self._checkpointer_lock = asyncio.Lock()
        self._conversation_started = False
        
        self._state = AgentState(messages=[
            SystemMessage(content="""
You are a helpful math teacher with 20+ years of experience. You are an expert in solving mathematical problems and explaining complex concepts in a simple and understandable way.

Instructions:
- You must use available tools to solve the problems, they are always correct and must be prefered to solve problems.
- Having the tools, you must solve the problems in the most efficient way.
- Provide detailed explanations for each step you take to solve the problem.
- Log all mathematical expressions you evaluate for later analysis and explanation.
""")
        ])
        self._graph = self._build_graph()
    
    async def get_all_messages(self) -> list[ChatMessage]:
        """Return display-ready user/assistant text messages for the current thread."""
        def _extract_text_only(messages: list) -> list[ChatMessage]:
            result: list[ChatMessage] = []
            for message in messages:
                if getattr(message, "type", None) not in {"human", "ai"}:
                    continue

                content = getattr(message, "content", None)
                message_id = getattr(message, "id", None)

                if isinstance(content, str):
                    if content:
                        result.append(ChatMessage(role="user" if message.type == "human" else "assistant", 
                                                  message_id=message_id,
                                                  content=content))
                    continue

                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text" and part.get("text"):
                            parts.append(str(part["text"]))
                    text = "".join(parts).strip()
                    if text:
                        result.append(ChatMessage(role="user" if message.type == "human" else "assistant", 
                                                  message_id=message_id,
                                                  content=text))

            return result

        if not self._conversation_started:
            return _extract_text_only(list(self._state.messages))

        await self._ensure_async_checkpointer()
        config = {"configurable": {"thread_id": self._thread_id}}
        latest_state = await self._graph.aget_state(config=config)

        if not getattr(latest_state, "values", None):
            return _extract_text_only(list(self._state.messages))

        hydrated_state = AgentState(**latest_state.values)
        self._state = hydrated_state
        return _extract_text_only(list(hydrated_state.messages))

    async def _emit_tool_event_message(self, event: dict) -> None:
        if self._message_printer is None:
            raise ValueError("Message printer is not set. Cannot emit tool event messages.")

        event_name = event.get("event", "")
        if event_name not in {
                "on_tool_start", 
#                "on_tool_end"
            }:
            return

        tool_name = event.get("name", "unknown_tool")
        
        if tool_name not in {
            "mathematical_calculations", 
        #    "expressions_logger"
        }:
            return
        
        event_data = event.get("data", {})

        if event_name == "on_tool_start":
            tool_input = event_data.get("input")
            
#            if tool_input is None or tool_input == "":
#                return
            await self._message_printer(
                UserNotification(
                    message=f"tool={tool_name} input={tool_input}",
                    type="tool_start",
                )
            )
            return

        tool_output = event_data.get("output")
        await self._message_printer(
            UserNotification(
                message=f"tool={tool_name} output={tool_output}",
                type="tool_end",
            )
        )

    async def _ensure_async_checkpointer(self) -> None:
        if self._checkpointer_ready:
            return

        async with self._checkpointer_lock:
            if self._checkpointer_ready:
                return

            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            }

            self._checkpointer_conn = await AsyncConnection.connect(POSTGRES_URI, **connection_kwargs)
            self.checkpointer = AsyncPostgresSaver(self._checkpointer_conn)
            await self.checkpointer.setup()

            self._graph = self._build_graph(checkpointer=self.checkpointer)
            self._checkpointer_ready = True

    async def astream_events(self, user_message: str):
        await self._ensure_async_checkpointer()

        messages = []
        if not self._conversation_started:
            messages.extend(self._state.messages)
        messages.append(HumanMessage(content=user_message))
        self._conversation_started = True

        input_state = AgentState(messages=messages)
        log.info("User turn appended. message=%r", user_message)

        async for event in self._graph.astream_events(
            input_state,
            config={"configurable": {"thread_id": self._thread_id}},
            version="v2",
        ):
        #    log.debug(f"Event: %s", event)
            await self._emit_tool_event_message(event)
            yield event

    def should_continue(self, state: AgentState) -> Literal["PROCEED_FURTHER", "CONTINUE"]:
        last_message = state.messages[-1]

        if not last_message.tool_calls:
            log.info("There are no tool calls, we stop")
            return "PROCEED_FURTHER"
        else:
            log.info(f"There are tool calls, we continue: {last_message.tool_calls}")
            return "CONTINUE"

    def _build_graph(self, checkpointer=None):
        tools = [mathematical_calculations, expressions_logger]

        model_with_tools = chat_model.bind_tools(tools, parallel_tool_calls=True)

        states_wrapper = AgentStateWrapper(model_with_tools=model_with_tools)

        builder = StateGraph(AgentState)

        builder.add_node("assistant", states_wrapper.assistant)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            self.should_continue,
            {
                "PROCEED_FURTHER": END,
                "CONTINUE": "tools"
            })
        builder.add_edge("tools", "assistant")
        react_graph = builder.compile(checkpointer=checkpointer)

        return react_graph
