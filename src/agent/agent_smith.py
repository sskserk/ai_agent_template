import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .states import AgentStateWrapper
from .tools import NotebookTool, mathematical_calculations


class AgentSmith:
    def __init__(self, notebook_tool: NotebookTool, chat_model: Any):
        self._notebook_tool = notebook_tool
        self._chat_model = chat_model
        self._graph = self._build_graph()

    def invoke(self, command) -> AgentState:
        state = self._achieve_goal(command=command)
        return state

    def _achieve_goal(self, command) -> AgentState:
        messages = [
            SystemMessage(content=f"""
You are a helpful math teacher with 20+ years of experience. You are an expert in solving mathematical problems and explaining complex concepts in a simple and understandable way.

Instructions:
- You must use available tools to solve the problems, they are always correct and must be prefered to solve problems.
- Having the tools, you must solve the problems in the most efficient way.
- Provide detailed explanations for each step you take to solve the problem.

Important: all mathematical operations must be summarized and stored in the notebook for later analyses by the teacher.
"""),
            AIMessage(content="""Sure, let me help students to solve math problem."""),
            HumanMessage(content=command),
        ]

        state = AgentState(messages=messages)
        thread_id = state.messages[0].content + "_" + datetime.datetime.now().isoformat()
        solve_state = self._graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        return AgentState(**solve_state)

    def should_continue(self, state: AgentState) -> Literal["PROCEED_FURTHER", "CONTINUE"]:
        last_message = state.messages[-1]

        if not last_message.tool_calls:
            print("There are no tool calls, we stop")
            return "PROCEED_FURTHER"

        print(f"There are tool calls, we continue: {last_message.tool_calls}")
        return "CONTINUE"

    def _build_graph(self):
        tools = [
            mathematical_calculations,
            self._notebook_tool.as_tool(),
        ]

        model_with_tools = self._chat_model.bind_tools(tools, parallel_tool_calls=True)
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
                "CONTINUE": "tools",
            },
        )
        builder.add_edge("tools", "assistant")

        return builder.compile()
