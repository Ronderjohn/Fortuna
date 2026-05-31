from fortuna.telegram.eval_suite import default_eval_fixture_path, evaluate_parse, load_eval_cases

_PARSE_CASES = [
    case for case in load_eval_cases(default_eval_fixture_path()) if case.layer == "parse"
]


def test_parser_eval_fixtures_cover_core_commands():
    ids = {case.id for case in _PARSE_CASES}
    assert "parse_search_equity" in ids
    assert "parse_analyze_future" in ids
    assert "parse_analyze_option" in ids


def test_parser_eval_cases():
    for case in _PARSE_CASES:
        evaluate_parse(case)
