from sympy import sympify, simplify
import re

def normalize_expression(expr):
    if "=" in expr:
        expr = expr.split("=")[-1]  # take RHS
    return expr.strip()

def validate_equation(student_expr, correct_expr):
    try:
        student_expr = normalize_expression(student_expr)
        correct_expr = normalize_expression(correct_expr)

        # Extract variable names dynamically
        vars_student = set(re.findall(r'[a-zA-Z]+', student_expr))
        vars_correct = set(re.findall(r'[a-zA-Z]+', correct_expr))
        all_vars = vars_student.union(vars_correct)

        # Create symbols dynamically
        local_dict = {var: sympify(var) for var in all_vars}

        student = simplify(sympify(student_expr, locals=local_dict))
        correct = simplify(sympify(correct_expr, locals=local_dict))

        return simplify(student - correct) == 0

    except Exception as e:
        print("Math error:", e)
        return False