"""Tests for isolated bounded conversation memory."""

from __future__ import annotations

import unittest

from normative_repository.agent import ConversationMemoryStore


class ConversationMemoryTests(unittest.TestCase):
    def test_context_is_partitioned_by_tenant_user_and_session(self) -> None:
        memory = ConversationMemoryStore(max_sessions=2)
        memory.add_user_fact("User fact", organization_id="org-a", user_id="user-a", session_id="session-a")
        memory.add_question("Question one", organization_id="org-a", user_id="user-a", session_id="session-a")
        same = memory.get(organization_id="org-a", user_id="user-a", session_id="session-a")
        isolated = memory.get(organization_id="org-b", user_id="user-a", session_id="session-a")
        self.assertEqual(same.user_facts, ["User fact"])
        self.assertEqual(same.prior_questions, ["Question one"])
        self.assertEqual(isolated.user_facts, [])

    def test_memory_is_bounded_and_explicitly_deletable(self) -> None:
        memory = ConversationMemoryStore(max_sessions=1)
        memory.add_user_fact("fact", organization_id="org", user_id="user", session_id="one")
        with self.assertRaises(ValueError):
            memory.add_question("question", organization_id="org", user_id="user", session_id="two")
        self.assertTrue(memory.delete(organization_id="org", user_id="user", session_id="one"))
        self.assertFalse(memory.delete(organization_id="org", user_id="user", session_id="one"))


if __name__ == "__main__":
    unittest.main()
