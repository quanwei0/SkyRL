import re

from skyrl_gym.envs.search.utils import normalize_answer, em_check, extract_solution


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_valid_sequence(text: str) -> tuple[bool, str]:
    assistant_match = re.search(r"<\|im_start\|>assistant\s*", text)
    content = text[assistant_match.end():] if assistant_match else text

    for tag in ["think", "search", "information", "answer"]:
        if len(re.findall(f"<{tag}>", content)) != len(re.findall(f"</{tag}>", content)):
            return False, f"Mismatched {tag} tags"

    parts = re.split(r"(</?(?:think|search|information|answer)>)", content)
    state = "start"
    transitions = {
        ("<think>",        ("start", "information")): "in_think",
        ("</think>",       ("in_think",)):             "after_think",
        ("<search>",       ("after_think",)):           "in_search",
        ("</search>",      ("in_search",)):             "after_search",
        ("<information>",  ("after_search",)):          "in_information",
        ("</information>", ("in_information",)):        "information",
        ("<answer>",       ("after_think",)):           "in_answer",
        ("</answer>",      ("in_answer",)):             "end",
    }

    for part in parts:
        if not part.strip():
            continue
        if re.match(r"</?(?:think|search|information|answer)>", part):
            next_state = next(
                (v for (tag, valid_states), v in transitions.items()
                 if tag == part and state in valid_states),
                None,
            )
            if next_state is None:
                return False, f"Unexpected tag {part} in state {state}"
            state = next_state
        else:
            if state not in ("in_think", "in_search", "in_information", "in_answer"):
                if part.strip():
                    return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
    return True, "Valid sequence format"


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    for block in extract_information_blocks(text):
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(block):
                return True
    return False


def compute_format_reward(solution_str: str, score: float = 1.0, fail_score: float = 0.0) -> float:
    """Returns score if the sequence has valid tag structure, else fail_score."""
    is_valid, _ = is_valid_sequence(solution_str)
    return score if is_valid else fail_score


def compute_retrieval_reward(solution_str: str, ground_truth: dict, score: float = 1.0, fail_score: float = 0.0) -> float:
    """Returns score if any <information> block contains the ground truth answer, else fail_score."""
    return score if is_retrieval_correct(solution_str, ground_truth["target"]) else fail_score


def compute_answer_reward(solution_str: str, ground_truth: dict, score: float = 1.0, fail_score: float = 0.0) -> float:
    """Returns score if the extracted <answer> matches ground truth via EM, else fail_score."""
    answer = extract_solution(solution_str)
    if answer is None:
        return fail_score
    return score if em_check(answer, ground_truth["target"]) else fail_score
