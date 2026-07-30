import re
import unittest

from conversation_state import ConversationStateMachine, empty_state


def subject(name, subject_type, unit_id=None):
    return {
        "unit_id": unit_id or name.lower().replace(" ", "-"),
        "name": name,
        "subject_type": subject_type,
        "title": "Projects" if subject_type == "project" else "People",
        "source_path": "projects.txt" if subject_type == "project" else "people.txt",
    }


class ConversationStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = ConversationStateMachine(rewrite_callable=self.fake_rewrite)
        self.rail = subject("Cape Cod Rail Resilience Project", "project")
        self.c3i = subject("Climate Careers Curricula Initiative", "project")
        self.tim = subject("Tim Cronin", "person")
        self.hannah = subject("Nyingilanyeofori Hannah Brown", "person")

    @staticmethod
    def fake_rewrite(message, subject):
        name = subject["name"]
        rewritten = message
        replacements = (
            (r"\b(the\s+)?(former|latter|first one|second one)\b", name),
            (r"\b(the\s+)?(other|previous)\s+(one|project|initiative|person)\b", name),
            (r"\b(that|this)\s+(project|initiative|program|person|one)\b", name),
            (r"\b(its|his|hers|their)\b", f"{name}'s"),
            (r"\b(it|she|he|him|her|they|them)\b", name),
        )
        for pattern, replacement in replacements:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        return rewritten if rewritten != message else f"Regarding {name}, {message.strip()}"

    def focus(self, active):
        state = empty_state()
        state.update({"mode": "focused", "active_subject": active, "candidate_subjects": [active]})
        return state

    def test_project_pronoun_rewrites_to_project(self):
        result = self.machine.resolve(
            "What specifically caused it to be launched?", self.focus(self.rail), []
        )
        self.assertTrue(result["resolved"])
        self.assertTrue(result["used_context"])
        self.assertIn(self.rail["name"], result["rewritten_query"])
        self.assertEqual(result["active_subject"]["subject_type"], "project")
        self.assertEqual(result["intent"], "cause")

    def test_person_pronoun_rewrites_to_person(self):
        result = self.machine.resolve(
            "What degree is she pursuing and where?", self.focus(self.hannah), []
        )
        self.assertTrue(result["resolved"])
        self.assertIn(self.hannah["name"], result["rewritten_query"])
        self.assertEqual(result["intent"], "education")

    def test_person_details_do_not_replace_active_project(self):
        result = self.machine.resolve(
            "Who leads it?", self.focus(self.rail), []
        )
        self.assertEqual(result["active_subject"], self.rail)
        self.assertIn(self.rail["name"], result["rewritten_query"])

    def test_explicit_new_person_switches_from_project(self):
        result = self.machine.resolve(
            "What is Tim Cronin's background?", self.focus(self.rail), [self.tim]
        )
        self.assertEqual(result["active_subject"], self.tim)
        self.assertFalse(result["used_context"])

    def test_comparison_retains_candidates_and_clarifies_pronoun(self):
        comparison = self.machine.resolve(
            "Compare C3I and the rail project.", empty_state(), [self.c3i, self.rail]
        )
        self.assertEqual(comparison["state"]["mode"], "comparing")
        follow_up = self.machine.resolve("What does it do?", comparison["state"], [])
        self.assertTrue(follow_up["needs_clarification"])
        self.assertEqual(follow_up["clarifying_question"], "Which project are you asking about?")
        self.assertEqual(len(follow_up["clarification_options"]), 2)

    def test_clarification_selection_resumes_pending_question(self):
        comparison = self.machine.resolve("Compare them.", empty_state(), [self.c3i, self.rail])
        clarification = self.machine.resolve("What year was it launched?", comparison["state"], [])
        selected = self.machine.resolve("the rail project", clarification["state"], [self.rail])
        self.assertEqual(selected["active_subject"], self.rail)

    def test_ordinal_selection_resumes_pending_question(self):
        comparison = self.machine.resolve("Compare them.", empty_state(), [self.c3i, self.rail])
        clarification = self.machine.resolve("What does it do?", comparison["state"], [])
        selected = self.machine.resolve("the second one", clarification["state"], [])
        self.assertTrue(selected["resolved"])
        self.assertEqual(selected["active_subject"], self.rail)
        self.assertIn(self.rail["name"], selected["rewritten_query"])

    def test_incompatible_person_pronoun_does_not_select_project(self):
        result = self.machine.resolve("What degree is she pursuing?", self.focus(self.rail), [])
        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["clarifying_question"], "Which person are you asking about?")
        self.assertEqual(result["clarification_options"], [])

    def test_standalone_topic_does_not_inherit_stale_subject(self):
        result = self.machine.resolve("Which students are currently at SSL?", self.focus(self.rail), [])
        self.assertFalse(result["resolved"])
        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["rewritten_query"], "Which students are currently at SSL?")

    def test_scope_reference_rewrites_without_inventing_person(self):
        state = empty_state()
        state.update({
            "mode": "scoped",
            "active_scope": {"name": "SSL Board of Directors", "title": "BoardOfDirectors"},
        })
        result = self.machine.resolve("Who on it works in policy?", state, [])
        self.assertTrue(result["scope_context"])
        self.assertIn("SSL Board of Directors", result["rewritten_query"])
        self.assertIsNone(result["state"]["active_subject"])

    def comparison_state(self):
        return self.machine.resolve(
            "Compare C3I and rail.", empty_state(), [self.c3i, self.rail]
        )["state"]

    def test_named_clarification_choice_resumes_pending_intent(self):
        clarification = self.machine.resolve(
            "What year was it launched?", self.comparison_state(), []
        )
        selected = self.machine.resolve(
            "Cape Cod Rail Resilience Project.", clarification["state"], [self.rail]
        )
        self.assertEqual(selected["active_subject"], self.rail)
        self.assertIn("What year", selected["rewritten_query"])
        self.assertNotEqual(selected["rewritten_query"], "Cape Cod Rail Resilience Project.")

    def test_correction_resumes_pending_clarification(self):
        clarification = self.machine.resolve(
            "What year was it launched?", self.comparison_state(), []
        )
        selected = self.machine.resolve("I meant C3I.", clarification["state"], [self.c3i])
        self.assertEqual(selected["active_subject"], self.c3i)
        self.assertIn("What year", selected["rewritten_query"])

    def test_former_and_latter_resolve_in_comparison_order(self):
        former = self.machine.resolve("What about the former's goals?", self.comparison_state(), [])
        latter = self.machine.resolve("What about the latter's funding?", self.comparison_state(), [])
        self.assertEqual(former["active_subject"], self.c3i)
        self.assertEqual(latter["active_subject"], self.rail)
        self.assertNotIn("former", former["rewritten_query"].lower())
        self.assertNotIn("latter", latter["rewritten_query"].lower())

    def test_plural_comparison_followups_keep_both_subjects(self):
        for question in (
            "What do they have in common?",
            "How do their goals differ?",
            "Which one was launched first?",
            "Which started earlier?",
            "Who leads each?",
        ):
            with self.subTest(question=question):
                result = self.machine.resolve(question, self.comparison_state(), [])
                self.assertFalse(result["needs_clarification"])
                self.assertTrue(result["comparison_context"])
                self.assertIn(self.c3i["name"], result["rewritten_query"])
                self.assertIn(self.rail["name"], result["rewritten_query"])

    def test_can_return_to_older_type_compatible_subject(self):
        c3i_state = self.machine.resolve("Tell me about C3I.", empty_state(), [self.c3i])["state"]
        tim_state = self.machine.resolve("Who is Tim Cronin?", c3i_state, [self.tim])["state"]
        project_return = self.machine.resolve(
            "Going back to the initiative, who funds it?", tim_state, []
        )
        self.assertEqual(project_return["active_subject"], self.c3i)

        tim_first = self.machine.resolve("Who is Tim Cronin?", empty_state(), [self.tim])["state"]
        c3i_second = self.machine.resolve("Tell me about C3I.", tim_first, [self.c3i])["state"]
        person_return = self.machine.resolve("What is his policy background?", c3i_second, [])
        self.assertEqual(person_return["active_subject"], self.tim)

    def test_failed_incompatible_reference_preserves_valid_subject(self):
        tim_state = self.machine.resolve("Who is Tim Cronin?", empty_state(), [self.tim])["state"]
        failed = self.machine.resolve("What does that project do?", tim_state, [])
        self.assertTrue(failed["needs_clarification"])
        recovered = self.machine.resolve("Actually, what is his role?", failed["state"], [])
        self.assertTrue(recovered["resolved"])
        self.assertEqual(recovered["active_subject"], self.tim)

    def test_long_continuation_uses_active_subject(self):
        result = self.machine.resolve(
            "And what foundation provided the original funding for the program's launch?",
            self.focus(self.c3i),
            [],
        )
        self.assertTrue(result["resolved"])
        self.assertTrue(result["used_context"])
        self.assertIn(self.c3i["name"], result["rewritten_query"])

    def test_correction_and_parallel_ellipsis_reuse_previous_facet(self):
        c3i_state = self.machine.resolve("Tell me about C3I.", empty_state(), [self.c3i])["state"]
        funding = self.machine.resolve("What foundation funds it?", c3i_state, [])["state"]
        for correction in ("Actually, I meant the rail project.", "And the rail project?"):
            with self.subTest(correction=correction):
                switched = self.machine.resolve(correction, funding, [self.rail])
                self.assertEqual(switched["active_subject"], self.rail)
                self.assertEqual(switched["intent"], "funding")
                self.assertIn("fund", switched["rewritten_query"].lower())

    def test_same_question_for_new_subject_reuses_previous_facet(self):
        c3i_state = self.machine.resolve("Tell me about C3I.", empty_state(), [self.c3i])["state"]
        funding_state = self.machine.resolve("Who funds it?", c3i_state, [])["state"]
        switched = self.machine.resolve("Same question for the rail project.", funding_state, [self.rail])
        self.assertEqual(switched["active_subject"], self.rail)
        self.assertEqual(switched["intent"], "funding")
        self.assertIn("fund", switched["rewritten_query"].lower())

    def test_other_subject_returns_to_recent_compatible_alternative(self):
        compared = self.comparison_state()
        c3i_focused = self.machine.resolve("Tell me about the first one.", compared, [])["state"]
        other = self.machine.resolve("What about the other project?", c3i_focused, [])
        self.assertTrue(other["resolved"])
        self.assertEqual(other["active_subject"], self.rail)
        self.assertIn(self.rail["name"], other["rewritten_query"])
        self.assertNotIn("other project", other["rewritten_query"].lower())

    def test_correction_to_new_clarification_option_resumes_pending_query(self):
        forum = subject("Climate Adaptation Forum", "project")
        clarification = self.machine.resolve(
            "What year was it launched?", self.comparison_state(), []
        )
        corrected = self.machine.resolve(
            "No, I meant the Climate Adaptation Forum.", clarification["state"], [forum]
        )
        self.assertTrue(corrected["resolved"])
        self.assertEqual(corrected["active_subject"], forum)
        self.assertIn("What year", corrected["rewritten_query"])

    def test_funding_attribute_can_follow_a_non_project_subject(self):
        foundation = subject("Barr Foundation", "person", "topic:barr-foundation")
        result = self.machine.resolve(
            "How did SSL use those funds through the Summer Anti-Racism Research Funding?",
            self.focus(foundation),
            [],
        )
        self.assertTrue(result["resolved"])
        self.assertFalse(result["needs_clarification"])
        self.assertEqual(result["active_subject"], foundation)

    def test_named_program_starts_new_topic_instead_of_false_clarification(self):
        result = self.machine.resolve(
            "What was the research on the Massachusetts MVP program about?",
            self.focus(self.tim),
            [],
        )
        self.assertFalse(result["resolved"])
        self.assertFalse(result["needs_clarification"])
        self.assertEqual(
            result["rewritten_query"],
            "What was the research on the Massachusetts MVP program about?",
        )
        self.assertEqual(result["state"]["last_query"], result["rewritten_query"])

    def test_relative_that_is_not_replaced_with_active_subject(self):
        result = self.machine.resolve(
            "What is the East Boston study that VanDeVeer's team worked on?",
            self.focus(self.tim),
            [self.tim],
        )
        self.assertTrue(result["resolved"])
        self.assertEqual(
            result["rewritten_query"],
            "Regarding Tim Cronin, What is the East Boston study that VanDeVeer's team worked on?",
        )
        self.assertNotIn("Tim Cronin VanDeVeer", result["rewritten_query"])

    def test_former_and_latter_select_clarification_options(self):
        clarification = self.machine.resolve(
            "What year was it launched?", self.comparison_state(), []
        )
        former = self.machine.resolve("the former", clarification["state"], [])
        latter = self.machine.resolve("the latter", clarification["state"], [])
        self.assertEqual(former["active_subject"], self.c3i)
        self.assertEqual(latter["active_subject"], self.rail)

    def test_multi_entity_correction_selects_new_subject_not_comparison(self):
        c3i_state = self.machine.resolve("Tell me about C3I.", empty_state(), [self.c3i])["state"]
        funding_state = self.machine.resolve("Who funds it?", c3i_state, [])["state"]
        corrected = self.machine.resolve(
            "No, not C3I, I meant the rail project.",
            funding_state,
            [self.c3i, self.rail],
        )
        self.assertTrue(corrected["resolved"])
        self.assertEqual(corrected["active_subject"], self.rail)
        self.assertEqual(corrected["intent"], "funding")


if __name__ == "__main__":
    unittest.main()
