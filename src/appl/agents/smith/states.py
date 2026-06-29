import logging
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

log = logging.getLogger(__name__)
from .agent_state import AgentState


class AgentStateWrapper:
    def __init__(self, model_with_tools=None):
        self.model_with_tools = model_with_tools

    async def _invoke_streaming(
        self,
        payload: Any,
        *,
        previous_response_id: Optional[str] = None,
    ):
        """Prefer streaming token generation and rebuild the final message from chunks."""
        invoke_kwargs = {}
        if previous_response_id is not None:
            invoke_kwargs["previous_response_id"] = previous_response_id

        aggregated_chunk = None
        async for chunk in self.model_with_tools.astream(payload, **invoke_kwargs):
            aggregated_chunk = chunk if aggregated_chunk is None else aggregated_chunk + chunk

        if aggregated_chunk is not None:
            if hasattr(aggregated_chunk, "to_message"):
                return aggregated_chunk.to_message()
            return aggregated_chunk

        # Fallback for providers/configurations that do not emit streamed chunks.
        return await self.model_with_tools.ainvoke(payload, **invoke_kwargs)

    def _messages_after_previous_response(self, state: AgentState) -> list:
        """Return all messages added after the AI response identified by previous_response_id."""
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if isinstance(msg, AIMessage):
                response_id = (msg.response_metadata or {}).get("id")
                if response_id == state.previous_response_id:
                    return list(state.messages[i + 1:])
        return list(state.messages)

    async def assistant(self, state: AgentState) -> AgentState:
        log.debug("Entering assistant, previous_response_id=%s", state.previous_response_id)

        new_message = None

        if state.previous_response_id is not None:
            messages_since = self._messages_after_previous_response(state)

            if messages_since and all(isinstance(m, ToolMessage) for m in messages_since):
                # Tool-call continuation: send only the tool responses, server holds full context.
                tool_messages = [
                    ToolMessage(content=m.content, tool_call_id=m.tool_call_id)
                    for m in messages_since
                ]
                log.debug(
                    "Tool continuation: sending %d tool messages with previous_response_id=%s",
                    len(tool_messages),
                    state.previous_response_id,
                )
                new_message = await self._invoke_streaming(
                    tool_messages,
                    previous_response_id=state.previous_response_id,
                )
            else:
                # New human turn: send only the latest human message content.
                last_human = next(
                    (m for m in reversed(state.messages) if isinstance(m, HumanMessage)),
                    None,
                )
                if last_human is None:
                    # With previous_response_id set, we must not send full history or non-human payloads.
                    log.warning(
                        "previous_response_id is set but no HumanMessage found; skipping model call"
                    )
                    return state

                payload = last_human.content
                log.debug(
                    "New human turn: sending last human message with previous_response_id=%s",
                    state.previous_response_id,
                )
                new_message = await self._invoke_streaming(
                    payload,
                    previous_response_id=state.previous_response_id,
                )
        else:
            # First turn: send full message history.
            log.debug("First turn: sending full message history (%d messages)", len(state.messages))
            new_message = await self._invoke_streaming(state.messages)

        if new_message is not None:
            state.messages.append(new_message)
            response_id = (getattr(new_message, "response_metadata", None) or {}).get("id")
            if isinstance(response_id, str) and response_id.startswith("resp_"):
                state.previous_response_id = response_id
                log.debug("Updated previous_response_id=%s", response_id)

        log.info("Total messages in state: %d", len(state.messages))
        return state