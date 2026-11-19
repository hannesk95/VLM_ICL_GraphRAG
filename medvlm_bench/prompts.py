"""Prompt construction for zero-shot, ICL and RAG conditions (forced-choice A/B answers)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    pos_text: str  # option for y = 1
    neg_text: str  # option for y = 0
    subject: str
    # --- wording that differs between tasks (the defaults are the brain / glioma texts, so brain prompts are unchanged)
    expert: str = "neuroradiologist"
    target: str = "the WHO CNS grade of the tumour"  # "Predict {target} from the imaging."
    question: str = "What is the WHO CNS grade of this tumour?"
    question_short: str = "Classify the glioma."  # prompt variant 1
    v2_role: str = "a careful neuro-oncology imaging assistant"
    v2_hint: str = ("Consider contrast enhancement, necrosis, peritumoral oedema, margins and "
                    "sequence-specific signal before deciding.")


GLIOMA_GRADE = TaskSpec(
    name="glioma_grade",
    pos_text="WHO CNS grade 4 (high grade)",
    neg_text="WHO CNS grade 2 or 3 (lower grade)",
    subject="pre-operative brain MRI of an adult patient with a diffuse glioma",
)

PROMPT_VARIANTS = (0, 1, 2)


def system_prompt(task: TaskSpec, layout_text: str, variant: int = 0) -> str:
    if variant == 0:
        return (f"You are an expert {task.expert}. You are shown {task.subject}. {layout_text} "
                f"Predict {task.target} from the imaging. Reply with a single letter.")
    if variant == 1:
        return f"{layout_text} Reply with a single letter."
    if variant == 2:
        return (f"You are {task.v2_role}. The images show {task.subject}. {layout_text} "
                f"{task.v2_hint} Reply with a single letter.")
    raise ValueError(variant)


def question_text(task: TaskSpec, variant: int, swap: bool) -> str:
    a, b = (task.neg_text, task.pos_text) if swap else (task.pos_text, task.neg_text)
    if variant == 1:
        return f"{task.question_short}\nA. {a}\nB. {b}\nAnswer with the letter only."
    return f"{task.question}\nA. {a}\nB. {b}\nAnswer:"


def label_letter(y: int, swap: bool) -> str:
    """Letter that is correct for label y given the option order."""
    return ("B" if y == 1 else "A") if swap else ("A" if y == 1 else "B")


def _user(images, text):
    return {"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]}


def _img_parts(images):
    return [{"type": "image", "image": im} for im in images]


def build_messages(task: TaskSpec, layout_text: str, query_images, shots=(), evidence: str | None = None,
                   query_findings_text: str = "", variant: int = 0, swap: bool = False, single_turn: bool = False):
    """shots: iterable of (images, label_y_used, findings_text). Returns the chat message list.

    single_turn=False: every example is its own user/assistant turn pair (multi-turn ICL).
    single_turn=True : all examples are interleaved inside ONE user message (the MedGemma model card
                       notes the model was not evaluated/optimised for multi-turn use, so both are benchmarked).
    """
    q = question_text(task, variant, swap)
    msgs = [{"role": "system", "content": [{"type": "text", "text": system_prompt(task, layout_text, variant)}]}]
    if single_turn and shots:
        parts = []
        for i, (imgs, y, ftxt) in enumerate(shots, 1):
            parts += [{"type": "text", "text": f"Example {i}." + (f" {ftxt}" if ftxt else "")}, *_img_parts(imgs),
                      {"type": "text", "text": f"{q}\n{label_letter(y, swap)}\n"}]
        head = "Now the case to classify." + (f" {query_findings_text}" if query_findings_text else "")
        tail = f"{head}\n" + (f"\nBackground knowledge retrieved for this case:\n{evidence}\n\n" if evidence else "")
        parts += [{"type": "text", "text": tail}, *_img_parts(query_images), {"type": "text", "text": q}]
        msgs.append({"role": "user", "content": parts})
        return msgs
    for i, (imgs, y, ftxt) in enumerate(shots, 1):
        head = f"Example {i}." + (f" {ftxt}" if ftxt else "")
        msgs.append(_user(imgs, f"{head}\n{q}"))
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": label_letter(y, swap)}]})
    head = ("Now the case to classify." if shots else "Case to classify.")
    if query_findings_text:
        head += f" {query_findings_text}"
    body = f"{head}\n"
    if evidence:
        body += f"\nBackground knowledge retrieved for this case:\n{evidence}\n\n"
    msgs.append(_user(query_images, body + q))
    return msgs


def n_images(messages) -> int:
    return sum(1 for m in messages for p in m["content"] if p["type"] == "image")


def yes_no_messages(layout_text: str, images, feature_question: str):
    """Single yes/no finding probe (A = yes, B = no)."""
    sys = ("You are an expert neuroradiologist. " + layout_text + " Answer the question about the imaging with a single letter.")
    q = f"Does the tumour show {feature_question}?\nA. Yes\nB. No\nAnswer:"
    return [{"role": "system", "content": [{"type": "text", "text": sys}]}, _user(images, q)]
