import json
import unittest

from tests.test_chatbot_regressions import OfflineChatbot


class FollowupStateIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.bot = OfflineChatbot()
        self.bot.entity_registry = [
            {
                "unit_id": "project-c3i", "title": "Projects",
                "source_path": "SEED_DOCUMENTS/Projects.txt", "source_url": "",
                "section_name": "Climate Careers Curricula Initiative (C3I)",
                "entity_type": "project", "detail_text": "C3I is a workforce development project.",
                "summary_text": "C3I will work with local and state policymakers to share findings on systemic barriers to blue and green jobs.",
            },
            {
                "unit_id": "project-rail", "title": "Projects",
                "source_path": "SEED_DOCUMENTS/Projects.txt", "source_url": "",
                "section_name": "Cape Cod Rail Resilience Project",
                "entity_type": "project",
                "detail_text": "The project was launched after a 300-foot rail embankment collapse in East Sandwich in 2020. Carlos Velasquez leads the project.",
                "summary_text": "",
            },
            {
                "unit_id": "project-cliir", "title": "Projects",
                "source_path": "SEED_DOCUMENTS/Projects.txt", "source_url": "",
                "section_name": "Climate Inequality and Integrative Resilience (CLIIR) Initiative",
                "entity_type": "project", "detail_text": "CLIIR studies climate inequality.",
                "summary_text": "",
            },
            {
                "unit_id": "project-ncjrc", "title": "Projects",
                "source_path": "SEED_DOCUMENTS/Projects.txt", "source_url": "",
                "section_name": "Northeast Climate Justice Research Collaborative",
                "entity_type": "project", "detail_text": "NCJRC is a research collaborative.",
                "summary_text": "",
            },
            {
                "unit_id": "person-carlos", "title": "Projects",
                "source_path": "SEED_DOCUMENTS/Projects.txt", "source_url": "",
                "section_name": "Carlos Velasquez", "entity_type": "person",
                "detail_text": "Carlos Velasquez is the rail project manager.", "summary_text": "",
            },
            {
                "unit_id": "person-tim", "title": "BoardOfDirectors",
                "source_path": "SEED_DOCUMENTS/BoardOfDirectors.txt", "source_url": "",
                "section_name": "Tim Cronin", "entity_type": "board_member",
                "detail_text": "Tim Cronin has a background in Boston climate and health policy.",
                "summary_text": "",
            },
            {
                "unit_id": "person-hannah", "title": "StudentsInterns",
                "source_path": "SEED_DOCUMENTS/StudentsInterns.txt", "source_url": "",
                "section_name": "Nyingilanyeofori Hannah Brown", "entity_type": "person",
                "detail_text": "She is a Ph.D. candidate in Global Governance and Human Security at the University of Massachusetts Boston.",
                "summary_text": "",
            },
        ]
        self.bot.document_registry = [
            {
                "title": "AnnualReport2021",
                "source_path": "SEED_DOCUMENTS/Annual Reports/AnnualReport2021.txt",
                "category": "Annual Reports", "folder_label": "Annual Reports",
            },
            {
                "title": "UMB-SSL-2025-Impact_Report",
                "source_path": "SEED_DOCUMENTS/Annual Reports/UMB-SSL-2025-Impact_Report.pdf",
                "category": "Annual Reports", "folder_label": "Annual Reports",
            },
        ]

    def stream_done_payload(self, question, history=None):
        events = list(self.bot.answer_stream(question, history))
        payloads = []
        for event in events:
            for line in event.splitlines():
                if line.startswith("data: "):
                    payloads.append(json.loads(line[6:]))
        return next(payload for payload in payloads if payload.get("done"))

    def test_streamed_shortcut_carries_project_state(self):
        payload = self.stream_done_payload("What is the Cape Cod Rail Resilience Project?")
        state = payload["conversation_state"]
        self.assertEqual(state["mode"], "focused")
        self.assertEqual(state["active_subject"]["unit_id"], "project-rail")

    def test_person_named_in_project_answer_does_not_steal_focus(self):
        result = {
            "reply": "Carlos Velasquez leads the Cape Cod Rail Resilience Project.",
            "response_mode": "rail_summary_shortcut",
            "needs_clarification": False,
        }
        state = self.bot.build_next_conversation_state(
            [], "What is the Cape Cod Rail Resilience Project?", result
        )
        self.assertEqual(state["active_subject"]["unit_id"], "project-rail")

    def test_rail_cause_followup_uses_rail_source(self):
        state = self.bot.build_next_conversation_state(
            [], "What is the Cape Cod Rail Resilience Project?", {"needs_clarification": False}
        )
        resolution = self.bot.resolve_conversation_turn(
            "What specifically caused it to be launched?",
            [{"user": "What is the Cape Cod Rail Resilience Project?", "assistant": "It is led by Carlos Velasquez.", "state": state}],
        )
        self.assertTrue(resolution["used_context"])
        self.assertIn("Cape Cod Rail Resilience Project", resolution["rewritten_query"])
        self.assertEqual(resolution["query_route"]["target_source_paths"], ["SEED_DOCUMENTS/Projects.txt"])

    def test_tim_background_followup_stays_on_tim(self):
        state = self.bot.build_next_conversation_state(
            [], "Who is Tim Cronin?", {"needs_clarification": False}
        )
        resolution = self.bot.resolve_conversation_turn(
            "What is his background in Boston climate and health policy?",
            [{"user": "Who is Tim Cronin?", "assistant": "Tim Cronin is an advisory board member.", "state": state}],
        )
        self.assertEqual(resolution["active_subject"]["unit_id"], "person-tim")
        self.assertEqual(resolution["query_route"]["target_source_paths"], ["SEED_DOCUMENTS/BoardOfDirectors.txt"])

    def test_hannah_three_turn_chain_keeps_person(self):
        state = self.bot.build_next_conversation_state(
            [], "Who is Nyingilanyeofori Hannah Brown?", {"needs_clarification": False}
        )
        history = [{
            "user": "Who is Nyingilanyeofori Hannah Brown?",
            "assistant": "She coordinates an SSL grant.", "state": state,
        }]
        background = self.bot.resolve_conversation_turn("What is her background?", history)
        history.append({"user": "What is her background?", "assistant": "She has policy experience.", "state": background["state"]})
        degree = self.bot.resolve_conversation_turn("What degree is she pursuing and where?", history)
        self.assertEqual(degree["active_subject"]["unit_id"], "person-hannah")
        self.assertIn("Nyingilanyeofori Hannah Brown", degree["rewritten_query"])

    def test_comparison_then_ambiguous_followup_offers_both_projects(self):
        question = "Compare the Climate Careers Curricula Initiative and Cape Cod Rail Resilience Project."
        state = self.bot.build_next_conversation_state(
            [], question, {"needs_clarification": False, "response_mode": "project_comparison"}
        )
        history = [{
            "user": question,
            "assistant": "C3I develops careers while the rail project improves infrastructure.",
            "state": state,
        }]
        clarification = self.bot.answer("What does it do?", history)
        self.assertTrue(clarification["needs_clarification"])
        self.assertEqual(clarification["response_mode"], "conversation_state_clarification")
        self.assertEqual(len(clarification["clarification_options"]), 2)

    def test_comparison_facet_followup_does_not_repeat_generic_summary(self):
        question = "Compare C3I and the Cape Cod Rail Resilience Project."
        initial = self.bot.answer(question)
        state = self.bot.build_next_conversation_state([], question, initial)
        history = [{"user": question, "assistant": initial["reply"], "state": state}]

        for prompt in ("Which started earlier?", "Who leads each?"):
            with self.subTest(prompt=prompt):
                follow_up = self.bot.answer(prompt, history)
                self.assertFalse(follow_up["needs_clarification"])
                self.assertNotEqual(follow_up.get("response_mode"), "project_comparison_shortcut")
                self.assertNotEqual(follow_up.get("response_mode"), "project_registry_guard")

    def test_explicit_person_switch_after_project_is_not_followup(self):
        state = self.bot.build_next_conversation_state(
            [], "What is the Cape Cod Rail Resilience Project?", {"needs_clarification": False}
        )
        resolution = self.bot.resolve_conversation_turn(
            "Who is Tim Cronin?",
            [{"user": "What is the rail project?", "assistant": "A resilience project.", "state": state}],
        )
        self.assertFalse(resolution["used_context"])
        self.assertEqual(resolution["active_subject"]["unit_id"], "person-tim")

    def test_standalone_students_query_does_not_inherit_project(self):
        state = self.bot.build_next_conversation_state(
            [], "What is the Cape Cod Rail Resilience Project?", {"needs_clarification": False}
        )
        resolution = self.bot.resolve_conversation_turn(
            "Which students are currently at SSL?",
            [{"user": "What is the rail project?", "assistant": "A resilience project.", "state": state}],
        )
        self.assertFalse(resolution.get("used_context", False))
        self.assertEqual(resolution["rewritten_query"], "Which students are currently at SSL?")

    def test_project_identity_requires_canonical_or_distinctive_evidence(self):
        c3i = self.bot.find_conversation_subject_entities(
            "Tell me about the Climate Careers Curricula Initiative."
        )
        self.assertEqual([entity["unit_id"] for entity in c3i], ["project-c3i"])

        cliir = self.bot.find_conversation_subject_entities("What is the CLIIR Initiative?")
        self.assertEqual([entity["unit_id"] for entity in cliir], ["project-cliir"])

        generic = self.bot.find_conversation_subject_entities(
            "What specific research project is she working on?"
        )
        self.assertEqual(generic, [])

        policy = self.bot.answer(
            "What does the C3I program say it will do with local and state policymakers?"
        )
        self.assertFalse(policy["needs_clarification"])
        self.assertIn("policymakers", policy["reply"])

    def test_named_non_registry_topics_become_source_scoped_subjects(self):
        report_scope = self.bot.detect_conversation_group_scope(
            "What honors were listed in the 2020-21 annual report?"
        )
        self.assertEqual(
            report_scope["source_path"],
            "SEED_DOCUMENTS/Annual Reports/AnnualReport2021.txt",
        )
        prior = {
            "version": 2, "mode": "scoped", "active_subject": None,
            "candidate_subjects": [], "active_scope": report_scope,
            "pending_query": None, "last_intent": None, "clarification_options": [],
        }
        maria = self.bot.resolve_conversation_turn(
            "Tell me about Maria Ivanova's appointment.",
            [{"user": "Which honors?", "assistant": "Several.", "state": prior}],
        )
        self.assertEqual(maria["active_subject"]["name"], "Maria Ivanova")
        self.assertEqual(maria["active_subject"]["subject_type"], "person")
        self.assertEqual(
            maria["query_route"]["target_source_paths"],
            ["SEED_DOCUMENTS/Annual Reports/AnnualReport2021.txt"],
        )

        event = self.bot.resolve_conversation_turn(
            "Tell me about the 'All We Can Save' event.",
            [{"user": "Which events?", "assistant": "Several.", "state": prior}],
        )
        self.assertEqual(event["active_subject"]["name"], "All We Can Save")
        self.assertEqual(event["active_subject"]["subject_type"], "entity")

        julie = self.bot.resolve_conversation_turn(
            "Who is Julie Wormser and what did she say in the 2020-21 annual report?",
            [],
        )
        self.assertEqual(julie["active_subject"]["name"], "Julie Wormser")
        self.assertEqual(
            julie["query_route"]["target_source_paths"],
            ["SEED_DOCUMENTS/Annual Reports/AnnualReport2021.txt"],
        )


if __name__ == "__main__":
    unittest.main()
