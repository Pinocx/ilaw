"""Tests for immutable prompt version management."""

from __future__ import annotations

import unittest

from normative_repository.llm import PromptRegistry, PromptTemplate, default_prompt_registry


class PromptRegistryTests(unittest.TestCase):
    def test_retrieves_and_renders_versioned_legal_prompt(self) -> None:
        registry = default_prompt_registry()
        prompt = registry.get("legal-research-answer", "v1")
        self.assertEqual(prompt.render(language="Spanish").split(".")[0], "You are a legal research analyst")
        self.assertEqual(registry.versions("legal-research-answer"), ["v1"])

    def test_registered_prompt_version_cannot_be_overwritten(self) -> None:
        registry = PromptRegistry()
        prompt = PromptTemplate(prompt_id="task", version="v1", system_template="Safe prompt")
        registry.register(prompt)
        with self.assertRaises(ValueError):
            registry.register(prompt)


if __name__ == "__main__":
    unittest.main()
