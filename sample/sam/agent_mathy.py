from langgraph.graph import START, StateGraph, END
from typing import List, TypedDict, Annotated, Optional, Literal
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from .state import MathProblemState, MathProblemDefinition
from appl.agents.agent.sam.states import SolveStateWrapper
from langgraph.prebuilt import ToolNode
from appl.agents.agent.sam.tools import (solve_equation,
                                        calculate_expression_value,
                                        calculate_equality_of_expressions,
                                        solve_inequality,
                                        )
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
import asyncio
from appl.models import ModelsManager
from appl.domain import (StudentProfile,
                         get_language_by_code,
                         ProblemToSolve,
                         ReasoningEvent,
                         SolveMessageEvent,
                         HumanChatMessage
                         )
from appl.integration import OCRResult
import json
from appl.process import Repository
from appl.persistence import DBStore
from appl.tools.i18n import _
import logging
import uuid


log = logging.getLogger(__name__)


class AgentMatt:
    def __init__(self, status_notification_callback = None, app_configuration = None, solve_id = -1):
        self._status_notification_callback = status_notification_callback
        self._current_step_number = 0
        self.states_wrapper = None
        self.solve_id = solve_id
        self.thread_id = None
        self.checkpointer = None
        self._checkpointer_conn = None
        self._checkpointer_ready = False
        self._checkpointer_lock = asyncio.Lock()
        self._app_configuration = app_configuration

        self._graph = self._build_graph()


    def get_graph_visualization(self) -> str:
        return self._graph.get_graph().draw_png()


    async def solve(self,
              problem_to_solve: ProblemToSolve,
              student_profile: StudentProfile,
              thread_id: str,
              ocr_result: OCRResult = None,
              solve_id: int = 0,
        ) -> str:

        self.solve_id = solve_id
        self.thread_id = thread_id
        self.states_wrapper.solve_id = solve_id
        human_message: HumanMessage = None

        human_message_id = f"hm_{str(uuid.uuid1())}"

        if problem_to_solve.image is None:
            # no image information has been attached
            message = f"""Please solve the problem: {problem_to_solve.content}.
Develop a step-by-step plan to solve the problem, then execute it, explaining each step in detail and enhancing your explanations with mathematical expressions and related known formulas (showing substitutions). Use the provided tools"""

            human_message = HumanMessage(content=message.strip(),
                                        id=human_message_id
                                        )
        #    log.debug(f"==============human_message.id: {human_message.id}")
        else:
            if problem_to_solve.image is not None:
                log.debug(f"User has provided image data")
                # image is provided as base64 sequence (e.g data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA.....)

            # image information has been provided. We need to include it in the message but don't forget about additional annotation
            message = None

            if problem_to_solve.content is not None and len(problem_to_solve.content) > 0:
                message = f"{problem_to_solve.content}. You are also provided with an image of the problem containing related content within the green rectangle. Focus on it, but also examine the entire image, as it may contain important details crucial to solving the problem."
            else:
                message = f"""Attached is an image outlining the problem. Focus on the content within the green rectangle, but also examine the entire image, as it may contain important details crucial to solving the problem."""

            message += """ Develop a step-by-step plan to solve the problem, then execute it, explaining each step in detail and enhancing your explanations with mathematical expressions and related known formulas (showing substitutions). Use the provided tools"""

            human_message = HumanMessage(content=[
                                    {"type": "text", "text": message.strip()},
                                    {"type": "image_url", "image_url": {"url": problem_to_solve.image.data}}
                                ],
                                id=human_message_id
                                )
        # if True:
        #     return "Sorry, I am currently unable to solve problems with images. Please try again with a text-only problem."

        state = await self._solve_problem(
                problem = MathProblemDefinition(problem=problem_to_solve.content),
                human_message=human_message,
                student_profile=student_profile
            )

        response = []

        # solve original problem and save the human message to the database, so we have a record of the original problem and the human message that triggered the solution process
        await self.save_human_solve_message(solve_id=self.solve_id,
                                        message_ref_id=human_message.id,
                                        content=problem_to_solve.content
                                            )

        for message in state.messages:
            if message.type not in ['ai']:
                continue

            if isinstance(message.content, list):
                for part in message.content:
                    if part.get("type") and part.get("type") == "text":
                        response.append(part.get("text"))
                    else:
                        response.append(part)
            else:
                response.append(message.content)

        return "\n".join(response)


    async def save_human_solve_message(self, solve_id: int, message_ref_id: str, content: str) -> str:
        id = await Repository().save_solve_message(
            solve_id=solve_id,
            message_ref_id=message_ref_id,
            content=content
        )
        return id

    async def get_solve_messages(self, solve_id: int) -> dict[str, HumanChatMessage]:
        messages = await Repository().get_solve_messages(solve_id=solve_id)

        # remap messages to dict with message_ref_id as key and HumanChatMessage as value
        messages_dict = {}
        for message in messages:
            if message.message_ref_id is not None:
                messages_dict[message.message_ref_id] = message

        return messages_dict


    async def continue_conversation(self,
                                    problem_to_solve: ProblemToSolve,
                                    student_profile: StudentProfile
        ) -> str:

        message_content = problem_to_solve.content
        human_message_id = f"hmc_{str(uuid.uuid1())}"

        human_message = HumanMessage(content= message_content.strip(), id=human_message_id)

        # get the last state of the conversation from the database using checkpointer, and continue the conversation from there

        # 1.1 Get state id by solve_id
        log.debug(f"Trying to get the last state of the conversation for solve_id [{problem_to_solve.solve_id}]")

        solve_details = await Repository().get_user_recent_exercise_details(user_id=student_profile.id, solve_id=problem_to_solve.solve_id)

        self.thread_id = solve_details.thread_id

        await self._ensure_async_checkpointer()

        config = {"configurable": {"thread_id": solve_details.thread_id }}
        log.debug(f"Using config for state retrieval: {config}")

        latest_solve_state = await self._graph.aget_state(config=config)


        new_state = await self._graph.ainvoke( {"messages": [human_message]}, state=latest_solve_state, config=config)

        await self.save_human_solve_message(solve_id=problem_to_solve.solve_id,
                                        message_ref_id=human_message.id,
                                        content=problem_to_solve.content
                                            )

        response_state = MathProblemState(**new_state)

        # Find the latest AI message
        latest_ai_message = None
        for message in reversed(response_state.messages):
            if message.type == 'ai':
                latest_ai_message = message
                break

        if latest_ai_message is None:
            return ""

        # Extract text from the latest AI message
        if isinstance(latest_ai_message.content, list):
            # Find the last text part in the content list
            text_parts = [part.get("text") for part in latest_ai_message.content if part.get("type") == "text"]
            return text_parts[-1] if text_parts else str(latest_ai_message.content)
        else:
            return str(latest_ai_message.content)


    async def get_state_messages(self, thread_id: str) -> List[AnyMessage]:
        await self._ensure_async_checkpointer()

        config = {"configurable": {"thread_id": thread_id }}

        latest_solve_state = await self._graph.aget_state(config=config)
        #log.debug(f"Got state details from the database for thread_id [{thread_id}]: {latest_solve_state}")
        solve_state = MathProblemState(**latest_solve_state.values)

        return solve_state.messages

    async def _solve_problem(self,
                       problem: MathProblemDefinition,
                       human_message: HumanMessage,
                       student_profile: StudentProfile,
        ) -> MathProblemState:

#
#- Tools may return reference codes; include these codes in your explanations for transparency as they contain math expressions.
#

        messages = [
            SystemMessage(content=f"""You are Mathy, a math teacher whose mission is to solve mathematical problems and to provide clear, detailed explanations at every stage.
Instructions:
- Analyze the math problem and outline a detailed step-by-step solution plan. For each step, specify what you will do and the reasoning behind your method or approach
- Clearly communicate your understanding of the problem, any provided options, and your planned approach to the student. Proceed directly with the solution; student approval is not required
- Use at least one tool when formulas or math are involved
- Perform calculations or comparisons with the provided tools—no manual calculations (do not mention tool use)
- Internally validate tool-based results before proceeding to the next step
- Present steps logically; use \\cancelto to visualize reductions (e.g., \\cancelto{2}{4}). Show all substitutions and transformations explicitly
- When incorporating answers from the tools into your solution, clearly restore and display the sequence of calculations to enhance the student's understanding
- If there is no solution or you cannot solve it, always explain why it cannot be solved. Until it is explicitly mentioned, do not solve problems in complex numbers
- For multiple-choice, solve, show work, confirm the available options, and select/record the matching option. Be very careful to select the correct option, sometimes options are very similar, so self reflect on the correctness of the choice. Also, share the reasoning for selecting the option
- Validate the final answer using an appropriate tool before presenting it to the student; do not offer next steps, add extra explanations, request additional information, or perform any calculations
- Present all mathematical expressions in LaTeX wrapped, avoiding arrays to ensure proper rendering. Use \(...\) for inline math and \[...\] for block math
- Tailor any explanations to the student grade level, as specified by "{student_profile.get_math_grade_level_name()}". Use the "{get_language_by_code(student_profile.exercise_locale)}" language for all responses, explanations, and units of measurement"""),
            human_message
        ]



        # messages = [
        #     # SystemMessage(content=f"""You are Sam, a math teacher whose mission is to solve mathematical problems and to provide clear, detailed explanations at every stage."""),
        #     HumanMessage(content=f"""say 'OK'"""),
        # ]

        state = MathProblemState(problem=problem, messages=messages, student_profile=student_profile)

        if not self.thread_id:
            raise ValueError("Thread ID must be provided to solve the problem")

        await self._ensure_async_checkpointer()
        solve_state = await self._graph.ainvoke(state, config = {"recursion_limit": 50, "configurable": { "thread_id": self.thread_id, "solve_id": self.solve_id}})

    #        log.debug(f"State thread_id: {self.thread_id} | solve_id: {self.solve_id} {solve_state}")

        return MathProblemState(**solve_state)

    async def _ensure_async_checkpointer(self):
        if self._checkpointer_ready:
            return

        async with self._checkpointer_lock:
            if self._checkpointer_ready:
                return

            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
                "options": "-c search_path=langgraph",
            }

            self._checkpointer_conn = await AsyncConnection.connect(DBStore.get_connection_dsn(), **connection_kwargs)

            self.checkpointer = AsyncPostgresSaver(self._checkpointer_conn)

            # Recompile graph with async checkpointer before first ainvoke.
            self._graph = self._build_graph(checkpointer=self.checkpointer)
            self._checkpointer_ready = True



    async def should_continue(self, state: MathProblemState) -> Literal["PROCEED_FURTHER", "CONTINUE"]:
        last_message = state.messages[-1]

        if not last_message.tool_calls:
            log.debug("There are no tool calls, stop")
            await self.send_reasoning_event("Preparing the final answer")

            return "PROCEED_FURTHER"

        else: # Otherwise if there is, we continue
            #tool_name = last_message.tool_calls[0]['name']
            self._current_step_number += 1

            message = _("Intermediate calculation %(step_number)d...", state.student_profile.exercise_locale) % {"step_number": self._current_step_number}

            await self.send_reasoning_event(message)

            log.debug(f"""Tool call number [{self._current_step_number}], we continue with:
************************************************
{json.dumps(last_message.tool_calls, indent=1)}
************************************************
                          """)

            # sample:
            # ************************************************
            # [
            #  {
            #   "name": "calculate_expression_value",
            #   "args": {
            #    "algebraic_expr": "-3 + 56 - 12"
            #   },
            #   "id": "call_eIgoyZlawUzwNNA4poPRH8VS",
            #   "type": "tool_call"
            #  }
            # ]
            # ************************************************


            return "CONTINUE"


    async def send_reasoning_event(self, message: str):
        await self._status_notification_callback(ReasoningEvent(content=message))

    async def send_solve_message_event(self, message: str):
        await self._status_notification_callback(SolveMessageEvent(solve_id=self.solve_id, role="assistant", content=message))


    def _build_graph(self, checkpointer = None):
        tools = [
#            reasoning_confirm_tool,
            #    calculate_limit,
                solve_inequality,
                solve_equation,
                calculate_equality_of_expressions,
                calculate_expression_value,
            ]

        model = ModelsManager.get_solver_model(use_responses_api=True)
        model_with_tools = model.bind_tools(tools, strict=True)

        builder = StateGraph(MathProblemState)

        self.states_wrapper = SolveStateWrapper(model_with_tools=model_with_tools,
                                                use_responses_api=True,
                                                notification_callback=self.send_solve_message_event,
                                                app_configuration=self._app_configuration
        )


        # Define nodes: these do the work
        builder.add_node("solution_category_detect", self.states_wrapper.solution_category_detect)
        builder.add_node("assistant", self.states_wrapper.assistant)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "solution_category_detect")
        builder.add_edge("solution_category_detect", "assistant")
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
