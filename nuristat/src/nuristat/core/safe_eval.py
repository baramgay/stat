"""사용자 수식 문자열 안전성 검증 — eval/df.eval 호출 전 던더 속성 접근 차단.

`().__class__.__base__.__subclasses__()` 류의 알려진 파이썬 샌드박스 우회는
`__builtins__` 제거만으로 막히지 않는다. 이 모듈은 eval에 넘기기 전 AST를
순회해 던더(``__x__``) 속성 접근을 거부한다.
"""

from __future__ import annotations

import ast

from nuristat.core.exceptions import UnsafeExpressionError


def validate_expression(expr: str) -> None:
    """expr을 파싱해 던더 속성 접근·import를 거부한다.

    Raises:
        UnsafeExpressionError: 금지된 패턴이 발견되거나 구문 오류인 경우.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"수식 구문 오류: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise UnsafeExpressionError(f"허용되지 않는 속성 접근입니다: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            raise UnsafeExpressionError(f"허용되지 않는 식별자입니다: {node.id}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeExpressionError("수식에는 import를 사용할 수 없습니다")
