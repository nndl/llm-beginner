"""Progressive-disclosure loader for SKILL.md capability packages."""
from __future__ import annotations

from pathlib import Path
import re

import yaml


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)

    def _documents(self):
        yield from sorted(self.skills_dir.glob("*/SKILL.md"))

    @staticmethod
    def _split(text: str) -> tuple[dict, str]:
        if not text.startswith("---\n"):
            return {}, text
        _, front_matter, body = text.split("---", 2)
        metadata = yaml.safe_load(front_matter) or {}
        return metadata, body.lstrip("\r\n")

    def list_skills(self) -> list[dict]:
        skills = []
        for document in self._documents():
            metadata, _ = self._split(document.read_text(encoding="utf-8"))
            if metadata.get("name") and metadata.get("description"):
                skills.append({"name": metadata["name"], "description": metadata["description"]})
        return skills

    def load(self, name: str) -> str:
        for document in self._documents():
            metadata, body = self._split(document.read_text(encoding="utf-8"))
            if metadata.get("name") == name:
                return body
        raise KeyError(f"Unknown skill: {name}")

    def match(self, task: str) -> list[dict]:
        """Match against metadata only; bodies are loaded explicitly afterwards."""
        words = set(re.findall(r"[a-z0-9_-]+", task.lower()))
        return [skill for skill in self.list_skills()
            if words.intersection(re.findall(r"[a-z0-9_-]+", skill["description"].lower()))]