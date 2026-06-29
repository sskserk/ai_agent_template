import logging
from appl.agents.agent.sam.state import MathProblemState, SolutionCategory
from langchain_core.messages import ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult
from typing import Any
import uuid


log = logging.getLogger(__name__)


class LexemPublisher:
    def __init__(self, notification_callback):
        self.notification_callback = notification_callback

        self.lexems = []
        self.prev_last_index = -1
        self.last_published_index = 0
        self.possible_formula_endings = [r"\)", r"\\)", r"\]", r"\\]"]
        self.possible_formula_starts = [r"\(", r"\\(", r"\[", r"\\["]

        self.formula_spotted = False
        self.formula_start_spotted = False


    async def publish_lexem(self, lexem: str) -> None:
        self.lexems.append(lexem)

        full_text = ''.join(self.lexems)

        # Check for the last occurrence of any possible formula ending
        last_index = -1
        for ending in self.possible_formula_endings:
            last_index = full_text.rfind(ending)
            if last_index > -1:
                self.formula_spotted = True
                break

        for start in self.possible_formula_starts:
            start_index = full_text.rfind(start)
            if start_index > -1 and (last_index == -1 or start_index > last_index):
                self.formula_start_spotted = True
                break

        is_break = '\n' in lexem or '\r' in lexem

#        log.debug(f"lexem [{lexem}], {is_break}, {self.formula_spotted}, {self.formula_start_spotted}, {last_index}, {self.formula_start_spotted}")

        if (last_index != -1 and last_index != self.prev_last_index) or (is_break and not self.formula_spotted and not self.formula_start_spotted):
            self.prev_last_index = last_index
            await self.publish()

    async def publish(self) -> None:
        message = ''.join(self.lexems[self.last_published_index:])

        await self.notification_callback(message=message)
        self.formula_spotted = False
        self.last_published_index = len(self.lexems)
        self.formula_start_spotted = False

    async def flush_unpublished(self) -> None:
        if self.last_published_index < len(self.lexems):
            await self.publish()

    async def get_content(self) -> str:
        return ''.join(self.lexems)


class QueueTokenHandler(AsyncCallbackHandler):
    """
    Push each new token into an asyncio.Queue so a consumer can read them
    as they arrive from the streaming LLM.
    """

    def __init__(self, notification_callback):
        self.chunks = []
        self.notification_callback = notification_callback
        self.formula_start_savepoint_index = -1
        self.last_sent_index = -1
        self.publisher = LexemPublisher(notification_callback)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if len(token) > 0:
            pass_token = token[0]
            try:
                if pass_token is not None and pass_token["type"] == "text":
                    #log.debug(f"Token: {pass_token} kwargs: {kwargs}")

                    message = pass_token["text"]
                    await self.publisher.publish_lexem(message)

                    #await self._trigger_status_message_send(message=message)

            except Exception as e:
                log.error(f"Error processing token: {e}")


    async def flush_undelivered_chunks(self):
        await self.publisher.flush_unpublished()

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        log.debug(f"END {response}, {kwargs}")

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        log.debug(f"Error: {error}, {kwargs}")

    async def get_content(self):
        return await self.publisher.get_content()


class SolveStateWrapper:
    def __init__(self,
                 model_with_tools: ChatOpenAI = None,
                 use_responses_api=True,
                 notification_callback=None,
                 app_configuration = None
    ):
        self.model_with_tools = model_with_tools
        self.use_responses_api = use_responses_api
        self.notification_callback = notification_callback
        self.app_configuration = app_configuration


    async def solution_category_detect(self, state: MathProblemState) -> MathProblemState:
        log.debug(f"Entering solution_category_detect method")

        state.problem.solution_category = SolutionCategory.CALC

        log.debug(f"Detected solution category: {state.problem.solution_category}")

        return state


    async def assistant(self, state: MathProblemState) -> MathProblemState:
        log.debug(f"Entering assistant method for previous_response_id [{getattr(state, 'previous_response_id', None)}], llm calls enabled [{self.app_configuration.is_llm_enabled if self.app_configuration else 'false'}]")

        streamed_request = False
        new_messages = None
        if state.previous_response_id is not None:

            tool_responses = []
            mark_found = False
            for mess in state.messages:
                if mark_found:
                    tool_responses.append(mess)

                if mess.response_metadata.get("id") == state.previous_response_id:
                    mark_found = True


            if tool_responses and all(tr.type == "tool" for tr in tool_responses):
                tool_messages = [
                    ToolMessage(
                        content=tr.content,
                        tool_call_id=tr.tool_call_id
                    )
                    for tr in tool_responses
                ]

                new_messages = await self.model_with_tools.ainvoke(
                    tool_messages,
                    previous_response_id=state.previous_response_id,
                    prompt_cache_key="shelper"
                )
            else:
                consumer = QueueTokenHandler(self.notification_callback)

                new_messages = None
                if self.app_configuration and self.app_configuration.is_llm_enabled:
                    new_messages = await self.model_with_tools.ainvoke(
                        state.messages[-1].content,
                        previous_response_id=state.previous_response_id,
                        stream=True,
                        config={"callbacks": [consumer]},
                        prompt_cache_key="shelper"
                    )
                else:
                    # llm calls are disabled, mock response with empty content message (.e.g OK)
                    mock_response_id = f"mocked_response_{uuid.uuid4()}"
                    new_messages = AIMessage(
                        content="OK continuation",
                        id=mock_response_id,
                        name="mocked_assistant",
                        additional_kwargs={
                            "refusal": None,
                        },
                        response_metadata={
                            "id": mock_response_id,
                            "model": "mocked-model",
                            "created": "2026-05-13T00:00:00Z",
                            "finish_reason": "stop",
                            "status": "completed",
                        },
                        tool_calls=[],
                        invalid_tool_calls=[],
                        usage_metadata={
                            "input_tokens": 32,
                            "output_tokens": 4,
                            "total_tokens": 36,
                        },
                    )
                await consumer.flush_undelivered_chunks()

        else:
            consumer = QueueTokenHandler(self.notification_callback)

            new_messages = None
            if self.app_configuration and self.app_configuration.is_llm_enabled:
                new_messages = await self.model_with_tools.ainvoke(state.messages, stream=True, config = {"callbacks": [consumer]}, prompt_cache_key="shelper")
            else:
                # llm calls are disabled, mock response with empty content message (.e.g OK)
                mock_response_id = f"mocked_response_{uuid.uuid4()}"
                new_messages = AIMessage(
                    content="OK",
                    id=mock_response_id,
                    name="mocked_assistant",
                    additional_kwargs={
                        "refusal": None,
                    },
                    response_metadata={
                        "id": mock_response_id,
                        "model": "mocked-model",
                        "created": "2026-05-13T00:00:00Z",
                        "finish_reason": "stop",
                        "status": "completed",
                    },
                    tool_calls=[],
                    invalid_tool_calls=[],
                    usage_metadata={
                        "input_tokens": 32,
                        "output_tokens": 4,
                        "total_tokens": 36,
                    },
                )
            await consumer.flush_undelivered_chunks()

            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Streaming response received: {new_messages}")
                content = await consumer.get_content()
                log.debug(f"result {content}")

        if not streamed_request:
            messages_to_append = []

            if isinstance(new_messages, list):

                messages_to_append = new_messages

                if self.use_responses_api:
                    state.previous_response_id = new_messages[-1].response_metadata["id"]
            else:
                log.info(f"Received single message from the assistant {new_messages}")
                log.debug(new_messages.response_metadata)

                if self.use_responses_api:
                    state.previous_response_id = new_messages.response_metadata["id"]

                messages_to_append = [new_messages]
                state.messages.append(new_messages)

            for message in messages_to_append:
                state.messages.append(message)

            log.info(f"Total messages in state: {len(state.messages)}")

            return state