import logging
from langgraph.graph import START, StateGraph, END
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from inta.states import AgentStateWrapper
from inta.state import AgentState
import datetime
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from sympy import latex
from sympy.parsing.latex import parse_latex


log = logging.getLogger(__name__)

chat_model = ChatOpenAI(
            model="gpt-5-mini-2025-08-07", 
            reasoning_effort="medium"
        )

    
@tool("mathematical_calculations", parse_docstring=True)
def mathematical_calculations(expression: str) -> str:
    """Performs mathematical calculations.

    Args:
        expression (str): A string containing the mathematical expression to be evaluated, written in LaTeX format.

    Returns:
        str: The result of the calculation in LaTeX format.
    """
    print(f"Evaluate expression:\n===================\n{expression}")
    try:

        expr1 = parse_latex(expression)
        print(f"Parsed expression: {expr1}")

        eval_result = expr1.simplify()
        
        result = latex(eval_result)
        print(f"========================Tool evaluation result: {result}")

        return result
    except Exception as e:
        print(f"Error solving equations: {e}")
        return e


class AgentTemplate:
    def __init__(self):
        self._graph = self._build_graph()

    def invoke(self, command ) -> str:
        state = self._achieve_goal(command=command)
        return state

    def _achieve_goal(self, command) -> str:

        messages = [
            SystemMessage(content=f"""
You are a helpful math teacher with 20+ years of experience. You are an expert in solving mathematical problems and explaining complex concepts in a simple and understandable way.

Instructions:
- You must use available tools to solve the problems, they are always correct and must be prefered to solve problems.
- Having the tools, you must solve the problems in the most efficient way.
- Provide detailed explanations for each step you take to solve the problem.
"""),
            AIMessage(content="""Sure, let me help students to solve math problem."""),
            HumanMessage(content=command),
        ]

        state = AgentState(messages=messages)

        thread_id = state.messages[0].content +"_" + datetime.datetime.now().isoformat()

        solve_state = self._graph.invoke(state, config = {"configurable": { "thread_id": thread_id }})

        return AgentState(**solve_state)


    def should_continue(self, state: AgentState) -> Literal["PROCEED_FURTHER", "CONTINUE"]:
        last_message = state.messages[-1]

        if not last_message.tool_calls:
            print("There are no tool calls, we stop")

            return "PROCEED_FURTHER"

        else: # Otherwise if there is, we continue
            print(f"There are tool calls, we continue: {last_message.tool_calls}")
            return "CONTINUE"


    def _build_graph(self):
        tools = [mathematical_calculations]

        model_with_tools = chat_model.bind_tools(tools, parallel_tool_calls=False)

        states_wrapper = AgentStateWrapper(model_with_tools=model_with_tools)

        builder = StateGraph(AgentState)

        # Define nodes: these do the work
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
        react_graph = builder.compile()

        return react_graph
    
    
math_agent = AgentTemplate() 
res = math_agent.invoke("""Calculate the 2+2*3.""")

print("Final response:==============")
for m in res.messages:
    print(f"Message. Type: {m.type}, Content: {m.content}")
