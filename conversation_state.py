"""Serializable conversation state transitions for corpus-grounded RAG chat."""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional


STATE_VERSION = 3


def empty_state() -> dict:
    return {
        "version": STATE_VERSION,
        "mode": "idle",
        "active_subject": None,
        "candidate_subjects": [],
        "active_scope": None,
        "pending_query": None,
        "last_intent": None,
        "clarification_options": [],
        "subject_history": [],
        "last_query": None,
    }


def normalize_state(state: Optional[dict]) -> dict:
    normalized = empty_state()
    if not isinstance(state, dict):
        return normalized
    for key in normalized:
        if key in state:
            normalized[key] = state[key]
    normalized["version"] = STATE_VERSION
    normalized["candidate_subjects"] = list(normalized.get("candidate_subjects") or [])
    normalized["clarification_options"] = list(normalized.get("clarification_options") or [])
    normalized["subject_history"] = unique_subjects(normalized.get("subject_history") or [])
    if normalized["mode"] == "focused" and not normalized.get("active_subject"):
        normalized["mode"] = "idle"
    if normalized["mode"] == "comparing" and len(normalized["candidate_subjects"]) < 2:
        normalized["mode"] = "focused" if normalized.get("active_subject") else "idle"
    if normalized["mode"] == "awaiting_clarification" and not normalized.get("pending_query"):
        normalized["mode"] = "focused" if normalized.get("active_subject") else "idle"
    return normalized


def subject_key(subject: dict) -> str:
    return str(subject.get("unit_id") or subject.get("name") or "").strip().lower()


def unique_subjects(subjects: Iterable[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for subject in subjects:
        key = subject_key(subject)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(subject)
    return result


class ConversationStateMachine:
    """Resolve discourse references without retrieving or generating an answer."""

    def __init__(
        self,
        rewrite_callable: Optional[Callable[[str, dict], str]] = None,
    ) -> None:
        self.rewrite_callable = rewrite_callable

    PERSON_MARKERS = (
        "person", "who is", "background", "biography", "bio", "degree",
        "university", "role", "title", "expertise", "research", "career",
    )
    PROJECT_MARKERS = (
        "project", "initiative", "program", "launched",
        "participants", "benefit", "purpose", "goal", "caused it",
        "what does it do", "what is it about", "who leads it",
    )
    CONTEXT_MARKERS = re.compile(
        r"\b(it|its|they|them|their|he|him|his|she|her|hers|this|that|those|these)\b",
        re.IGNORECASE,
    )
    ELLIPSIS_MARKERS = re.compile(
        r"^(and\s+)?(what|which|who|when|where|why|how|does|did|is|are|was|were|can|could)\b",
        re.IGNORECASE,
    )
    CONTINUATION_MARKERS = re.compile(
        r"^\s*(and\b|also\b|actually\b|instead\b|going back\b|back to\b|what about\b|how about\b|"
        r"(?:no[, ]+)?i mean(?:t)?\b|rather\b|same (?:question|thing)\b)",
        re.IGNORECASE,
    )

    def classify_intent(self, message: str) -> str:
        lowered = message.lower()
        intent_terms = (
            ("cause", ("what caused", "caused", "why was", "why did", "motivat", "launched in response")),
            ("time", ("what year", "when", "how long", "timeframe")),
            ("funding", ("fund", "grant", "supported by", "sponsor")),
            ("leadership", ("who leads", "leader", "director", "chair")),
            ("education", ("degree", "university", "college", "studying", "candidate")),
            ("background", ("background", "biography", "bio", "career")),
            ("role", ("role", "position", "title", "what does she do", "what does he do")),
            ("research", ("research", "expertise", "focus", "work on")),
            ("audience", ("who is it for", "who does it serve", "participants", "audience")),
            ("count", ("how many", "count", "number of")),
            ("summary", ("tell me more", "what does it do", "what is it about", "overview")),
        )
        for intent, markers in intent_terms:
            if any(marker in lowered for marker in markers):
                return intent
        return "fact"

    def expected_subject_type(self, message: str, prior: dict) -> Optional[str]:
        lowered = message.lower()
        active = prior.get("active_subject") or {}
        if active.get("subject_type") == "publication" and any(
            marker in lowered for marker in ("title", "author", "publication", "paper", "study")
        ):
            return "publication"
        if re.search(r"\b(she|her|hers|he|him|his|that person|this person)\b", lowered):
            return "person"
        if any(marker in lowered for marker in self.PROJECT_MARKERS):
            return "project"
        if any(marker in lowered for marker in self.PERSON_MARKERS):
            return "person"
        if self.CONTEXT_MARKERS.search(message) or self.ELLIPSIS_MARKERS.search(message.strip()):
            return active.get("subject_type") or None
        return None

    def needs_context(self, message: str, prior: dict) -> bool:
        if self.CONTEXT_MARKERS.search(message):
            return True
        if self.CONTINUATION_MARKERS.search(message):
            return True
        if re.search(
            r"\b(?:[A-Z][\w'-]*\s+){1,5}(?:program|project|initiative|study|grant|funding)\b",
            message,
        ):
            return False
        if re.search(r"\b(the|that|this)\s+(project|initiative|program|person)(?:'s|\b)", message, re.IGNORECASE):
            return True
        if re.search(
            r"\b(students?|interns?|alumni|fellows?|staff|board members?|projects?|publications?|affiliates?)\b",
            message,
            re.IGNORECASE,
        ):
            return False
        if prior.get("mode") in {"focused", "comparing", "awaiting_clarification", "scoped"}:
            stripped = message.strip()
            return len(stripped.split()) <= 10 and bool(self.ELLIPSIS_MARKERS.search(stripped))
        return False

    @staticmethod
    def compatible(subject: dict, expected_type: Optional[str]) -> bool:
        return not expected_type or subject.get("subject_type") == expected_type

    def select_clarification_subject(self, message: str, candidates: list[dict]) -> Optional[dict]:
        lowered = re.sub(r"[^a-z0-9\s]", " ", message.lower())
        if re.search(r"\b(first|1st|former)\b", lowered) and candidates:
            return candidates[0]
        if re.search(r"\b(second|2nd|two|last|latter)\b", lowered) and len(candidates) > 1:
            return candidates[1]
        matches: list[dict] = []
        for subject in candidates:
            name = str(subject.get("name", "")).lower()
            tokens = [token for token in re.findall(r"[a-z0-9]+", name) if len(token) > 2]
            if name and (name in message.lower() or (tokens and all(token in lowered for token in tokens[-2:]))):
                matches.append(subject)
        return matches[0] if len(matches) == 1 else None

    def rewrite(self, message: str, subject: dict) -> str:
        if not self.rewrite_callable:
            return message
        rewritten = self.rewrite_callable(message, subject)
        return str(rewritten).strip() or message

    def clarify(self, message: str, candidates: list[dict], expected_type: Optional[str], prior: Optional[dict] = None) -> dict:
        candidates = unique_subjects(candidates)[:4]
        label = "project" if expected_type == "project" else "person" if expected_type == "person" else "subject"
        options = [f"{subject.get('name')} ({subject.get('subject_type', 'subject')})" for subject in candidates]
        state = normalize_state(prior)
        history = unique_subjects(
            list(state.get("subject_history") or [])
            + ([state["active_subject"]] if state.get("active_subject") else [])
            + list(state.get("candidate_subjects") or [])
        )
        state.update({
            "mode": "awaiting_clarification",
            "candidate_subjects": candidates,
            "pending_query": message,
            "last_intent": self.classify_intent(message),
            "clarification_options": options,
            "subject_history": history,
        })
        return {
            "resolved": False,
            "needs_clarification": True,
            "clarifying_question": f"Which {label} are you asking about?",
            "clarification_options": options,
            "state": state,
        }

    def resolve(self, message: str, prior_state: Optional[dict], explicit_subjects: list[dict]) -> dict:
        prior = normalize_state(prior_state)
        explicit = unique_subjects(explicit_subjects)
        intent = self.classify_intent(message)

        if prior.get("mode") == "awaiting_clarification":
            candidate_keys = {subject_key(subject) for subject in prior["candidate_subjects"]}
            explicit_choice = next(
                (subject for subject in explicit if subject_key(subject) in candidate_keys),
                None,
            )
            selected = explicit_choice or self.select_clarification_subject(message, prior["candidate_subjects"])
            if not selected and len(explicit) == 1 and self.CONTINUATION_MARKERS.search(message):
                selected = explicit[0]
            if selected:
                pending = str(prior.get("pending_query") or "Tell me more.")
                pending_intent = str(prior.get("last_intent") or self.classify_intent(pending))
                return self._resolved(
                    self.rewrite(pending, selected), selected, pending_intent,
                    used_context=True, prior=prior,
                )

        comparison_candidates = unique_subjects(prior.get("candidate_subjects") or [])
        if prior.get("mode") == "comparing" and len(comparison_candidates) > 1:
            lowered = message.lower()
            if re.search(r"\b(former|first one|first project|first person)\b", lowered):
                selected = comparison_candidates[0]
                return self._resolved(self.rewrite(message, selected), selected, intent, used_context=True, prior=prior)
            if re.search(r"\b(latter|second one|second project|second person)\b", lowered):
                selected = comparison_candidates[1]
                return self._resolved(self.rewrite(message, selected), selected, intent, used_context=True, prior=prior)
            collective = bool(re.search(r"\b(they|them|their|both|each|common|differ|difference|compare)\b", lowered))
            comparative_selection = bool(re.search(
                r"\bwhich(?:\s+(?:one|project|initiative|person))?\b.*\b"
                r"(first|earlier|later|newer|older|more|less|most|least|larger|smaller|broader)\b",
                lowered,
            ))
            if collective or comparative_selection:
                names = " and ".join(str(subject.get("name", "")) for subject in comparison_candidates)
                state = normalize_state(prior)
                state.update({"last_intent": intent, "last_query": message, "pending_query": None})
                rewritten = self.rewrite(message, {
                    "name": names,
                    "subject_type": "comparison",
                })
                return {
                    "resolved": False, "needs_clarification": False,
                    "rewritten_query": rewritten,
                    "comparison_context": True, "used_context": True, "state": state,
                }

        if len(explicit) > 1:
            active = prior.get("active_subject") or {}
            correction = bool(re.match(r"^\s*(?:no\b|not\b|actually\b|i mean(?:t)?\b|rather\b|instead\b)", message, re.IGNORECASE))
            if active and correction and "compare" not in message.lower():
                selected = next(
                    (subject for subject in reversed(explicit) if subject_key(subject) != subject_key(active)),
                    None,
                )
                if selected:
                    previous = str(prior.get("last_query") or message)
                    rewritten = self.rewrite(previous, selected)
                    return self._resolved(
                        rewritten, selected,
                        str(prior.get("last_intent") or intent), used_context=True, prior=prior,
                    )
            state = normalize_state(prior)
            history = unique_subjects(
                list(state.get("subject_history") or [])
                + ([state["active_subject"]] if state.get("active_subject") else [])
                + explicit
            )[-12:]
            state.update({
                "mode": "comparing", "active_subject": None,
                "candidate_subjects": explicit, "pending_query": None,
                "last_intent": intent, "clarification_options": [],
                "subject_history": history, "last_query": message,
            })
            return {"resolved": False, "needs_clarification": False, "rewritten_query": message, "state": state}

        if len(explicit) == 1:
            subject = explicit[0]
            carry_previous = bool(
                prior.get("last_query")
                and self.CONTINUATION_MARKERS.search(message)
                and not re.search(r"\b(what|which|who|when|where|why|how|does|did|is|are|was|were|can|could)\b", message, re.IGNORECASE)
            )
            if carry_previous:
                previous = str(prior.get("last_query"))
                rewritten = self.rewrite(previous, subject)
                return self._resolved(
                    rewritten, subject, str(prior.get("last_intent") or intent),
                    used_context=True, prior=prior,
                )
            subject_name = str(subject.get("name", "")).strip()
            rewritten = (
                message
                if subject_name and subject_name.lower() in message.lower()
                else self.rewrite(message, subject)
            )
            return self._resolved(rewritten, subject, intent, used_context=False, prior=prior)

        if not self.needs_context(message, prior):
            state = normalize_state(prior)
            state["last_intent"] = intent
            state["last_query"] = message
            return {"resolved": False, "needs_clarification": False, "rewritten_query": message, "state": state}

        active_scope = prior.get("active_scope")
        if active_scope and not prior.get("active_subject") and not prior.get("candidate_subjects"):
            scope_name = str(active_scope.get("name", "")).strip()
            filter_text = str(active_scope.get("filter_text", "")).strip()
            rewritten = message
            if scope_name and self.rewrite_callable:
                rewritten = self.rewrite(message, {
                    "name": scope_name,
                    "subject_type": "scope",
                    "filter_text": filter_text,
                })
            state = normalize_state(prior)
            state["last_intent"] = intent
            return {
                "resolved": False, "needs_clarification": False,
                "rewritten_query": rewritten, "scope_context": True, "state": state,
            }

        expected_type = self.expected_subject_type(message, prior)
        candidates = unique_subjects(prior.get("candidate_subjects") or [])
        active = prior.get("active_subject")
        compatible = [subject for subject in candidates if self.compatible(subject, expected_type)]
        historical = [
            subject for subject in reversed(prior.get("subject_history") or [])
            if self.compatible(subject, expected_type)
        ]

        if active and re.search(r"\b(?:the\s+)?(?:other|previous)\s+(?:one|project|initiative|person)\b", message, re.IGNORECASE):
            alternative = next(
                (subject for subject in historical if subject_key(subject) != subject_key(active)),
                None,
            )
            if alternative:
                return self._resolved(
                    self.rewrite(message, alternative), alternative, intent,
                    used_context=True, prior=prior,
                )

        if active and self.compatible(active, expected_type):
            return self._resolved(self.rewrite(message, active), active, intent, used_context=True, prior=prior)
        if len(compatible) == 1:
            return self._resolved(self.rewrite(message, compatible[0]), compatible[0], intent, used_context=True, prior=prior)
        if len(compatible) > 1:
            return self.clarify(message, compatible, expected_type, prior)
        if historical:
            return self._resolved(self.rewrite(message, historical[0]), historical[0], intent, used_context=True, prior=prior)
        if active:
            return self.clarify(message, [], expected_type, prior)
        if candidates:
            return self.clarify(message, candidates, expected_type, prior)
        return {"resolved": False, "needs_clarification": False, "rewritten_query": message, "state": prior}

    def _resolved(
        self, rewritten: str, subject: dict, intent: str, *, used_context: bool,
        prior: Optional[dict] = None,
    ) -> dict:
        state = normalize_state(prior)
        history = unique_subjects(
            list(state.get("subject_history") or [])
            + ([state["active_subject"]] if state.get("active_subject") else [])
            + list(state.get("candidate_subjects") or [])
            + [subject]
        )[-12:]
        state.update({
            "mode": "focused", "active_subject": subject,
            "candidate_subjects": [subject], "last_intent": intent,
            "pending_query": None, "clarification_options": [],
            "subject_history": history, "last_query": rewritten,
        })
        return {
            "resolved": True, "needs_clarification": False,
            "rewritten_query": rewritten, "active_subject": subject,
            "intent": intent, "used_context": used_context, "state": state,
        }
