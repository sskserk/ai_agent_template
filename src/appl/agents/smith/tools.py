import logging

from langchain_core.tools import tool
from sympy import latex
from sympy.parsing.latex import parse_latex


log = logging.getLogger(__name__)


@tool("mathematical_calculations", parse_docstring=True)
def mathematical_calculations(expression: str) -> str:
    """Performs mathematical calculations.

    Args:
        expression (str): A string containing the mathematical expression to be evaluated, written in LaTeX format.

    Returns:
        str: The result of the calculation in LaTeX format.
    """
    log.debug(f"   Evaluate expression:\n===================\n{expression}")
    try:
        expr1 = parse_latex(expression)
        log.debug(f"    Parsed expression: {expr1}")

        eval_result = expr1.simplify()

        result = latex(eval_result)
        log.debug(f"    ========================Tool evaluation result: {result}")

        return result
    except Exception as e:
        log.error(f"Error solving equations: {e}")
        return str(e)


@tool("expressions_logger", parse_docstring=True)
def expressions_logger(expression: str) -> str:
    """Traces all tool calls for later analysis and explanation of the mathematical expression. Use it to log the expression that is being evaluated. Use this tool in parallel with the mathematical_calculations tool to log the expression being evaluated.

    Args:
        expression (str): A string containing the mathematical expression to be explained, written in LaTeX format.

    Returns:
        str: confirmation message that the expression has been stored for later analysis.
    """
    log.debug(f"   Traced expression:\n===================\n{expression}")
    try:
        log.debug("    ======================== OK")
        return "ok"
    except Exception as e:
        log.error(f"Error expressions_logger : {e}")
        return str(e)