from __future__ import annotations

import pytest
from support.conversation_eval_fakes import build_eval_assistant, build_eval_tools

from fortuna.telegram.eval_suite import (
    default_eval_fixture_path,
    evaluate_assistant,
    evaluate_parse,
    evaluate_route,
    evaluate_tool,
    load_eval_cases,
)

_EVAL_CASES = load_eval_cases(default_eval_fixture_path())
_PARSE_CASES = [case for case in _EVAL_CASES if case.layer == "parse"]
_TOOL_CASES = [case for case in _EVAL_CASES if case.layer == "tool"]
_ASSISTANT_CASES = [case for case in _EVAL_CASES if case.layer == "assistant"]


@pytest.mark.parametrize("case", _EVAL_CASES, ids=lambda case: case.id)
def test_conversation_eval_case(case, tmp_path):
    if case.layer == "parse":
        evaluate_parse(case)
    elif case.layer == "route":
        evaluate_route(case)
    elif case.layer == "tool":
        evaluate_tool(case, build_eval_tools(tmp_path))
    elif case.layer == "assistant":
        evaluate_assistant(case, build_eval_assistant(tmp_path))


@pytest.mark.parametrize("case", _PARSE_CASES, ids=lambda case: case.id)
def test_parser_eval_fixtures(case):
    evaluate_parse(case)
