import logging
from typing import List, Annotated, Optional
from appl.agents.agent.verifier.states import *
from appl.agents.agent.verifier import *
from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from .state import MathProblemState
from appl.tools import to_bool
from appl.agents.agent.e6nsolver import EquationsSolveAgent
from appl.agents.tooling.calc_expr_sympy import calculate_expression_tool_sympy
from appl.agents.tooling.expr_equal_sympy import compare_expressions_tool_sympy
from appl.agents.tooling.inequality import solve_inequality_tool

log = logging.getLogger(__name__)


@tool("solve_equation", parse_docstring=True)
def solve_equation(
    equations: List[str],
    vars: List[str],
    state: Annotated[MathProblemState, InjectedState]
    ) -> str:
    """Solve an equation or a system of equations

Args:
    equations (List[str]): The list of equations to solve, written in SymPy notation (e.g., [Eq(x**2, 4), Eq(2*cos(x)**2 + 5*sin(x), 4)])
    vars (List[str]): The list of variable names (as SymPy terms) to solve for (e.g., x, z, y_2)

Returns:
    str: A description of the solution to the equation or system of equations, including the values of the variables, written in SymPy notation"""

    log.debug(f"The [solve_equation_tool] is requested. Parameters [equations: {equations}, variables: {vars}]")
    for var in vars:
        if var.startswith("\\"):
            error_message = f"""Variable name '{var}' is not a valid identifier. Please use valid variable names consisting of letters, digits, or underscores, and starting with a letter (e.g., ["x", "y", "z", "x_{0}", "x_{1}", "y_{2}"])"""
            log.error(error_message)
            raise ValueError(error_message)

    try:
        agent = EquationsSolveAgent()
        solution = agent.solve(equations, vars)

        description = solution.describe()
        log.debug(f"Solution description: [{description}]")

        if not solution.is_empty():
            solutions = []
            for root_set in solution.root_sets:
                for root in root_set.roots:
                    solutions.append(root.variable + "=" + root.value + "")
            solutions_description = "\n".join(solutions)

            log.debug(f"Equations {equations} have been resolved into solutions: [{solutions_description}]")
            return solutions_description

        return description if description else "No solutions found"
    except Exception as e:
        log.error(f"Failed to solve equations '{equations}': {e}")
        raise ValueError(f"Failed to solve equations '{equations}': {e}")


@tool("calculate_expr", parse_docstring=True)
def calculate_expression_value(
    expr: str,
    caret: bool,
    state: Annotated[MathProblemState, InjectedState]
) -> str:
    """Calculate or simplify an algebraic expression; suitable for polynomials, rational expressions, bracket expansions, and set and interval operations, but not for pure constants

Args:
    expr (str): expression to calculate, written in SymPy notation (e.g., binomial(5, 2)+5*y, 3*x*(4+y)**2), limit(x**2 - 3*x, x, oo))
    caret (boolean): Interpret "^" as exponentiation

Returns:
    str: The evaluated value of the SymPy expression"""

    log.debug(f"The [calculate_expression_value] tool is requested. Parameters [{expr}, {caret}]")
    interpreted_caret = to_bool(caret)

    try:
        simplified_sympy = calculate_expression_tool_sympy(expr, is_caret=interpreted_caret)

        return simplified_sympy
    except Exception as e:
        log.error(f"Error evaluating expression '{expr}': {e}")
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


@tool("solve_inequality", parse_docstring=True)
def solve_inequality(inequality: str,
                     var: str,
                    state: Annotated[MathProblemState, InjectedState]
    ) -> str:
    """Solve a mathematical inequality for a specified variable using a CAS (computer algebra system)

    Args:
        inequality (str): The inequality to solve, written in SymPy notation (e.g., Le(x**2 + 2, 0), x/(x-1) >= 2).
        var (str): The variable to solve for, written in SymPy notation (e.g., x, y, x_1).

    Returns:
        str: A SymPy-formatted description of the values of the variable that satisfy the inequality.
    """

    try:
        log.debug(f"The [solve_inequality] is requested. Parameters [{inequality}, {var}]")

        solution = solve_inequality_tool(inequality, var)

        log.debug(f"The inequality [{inequality}] with symbol [{var}] is reduced to the [{solution}]")

        return solution
    except Exception as e:
        log.error(f"Failed to solve the inequality [{inequality}]: {e}")
        raise ValueError(f"Failed to solve the inequality [{inequality}]: {e}")


@tool("calculate_equality", parse_docstring=True)
def calculate_equality_of_expressions(
    left_expr: str,
    right_expr: str,
    state: Annotated[MathProblemState, InjectedState]
) -> Optional[int]:
    """Compare two mathematical expressions for equality or determine ordering. Returns 0 if they are equal, -1 if the left expression is less than the right, 1 if greater, or None if not comparable.

    Args:
        left_expr (str): Left-hand expression in SymPy notation (e.g., 2/3 + x*y**2)
        right_expr (str): Right-hand expression in SymPy notation (e.g., 2**2 + 5*x, Interval(0, 2, left_open=True))

    Returns:
        Optional[int]: Comparison result:
            0 if the expressions are mathematically equivalent,
            -1 if left_expr is strictly less than right_expr,
            1 if left_expr is strictly greater than right_expr,
            None if the expressions cannot be compared or the ordering is indeterminate.
    """

    log.debug(f"The [calculate_equality_of_expressions] tool is requested. Parameters [{left_expr}, {right_expr}]")

    try:
        result = compare_expressions_tool_sympy(left_expr, right_expr)
        log.debug(f"Compared expressions: left [{left_expr}], right [{right_expr}], result [{result}]")

        return result
    except Exception as e:
        log.error(f"Error evaluating expressions ['{left_expr}', '{right_expr}']. {e}")
        raise e