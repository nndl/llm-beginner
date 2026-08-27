"""Progressive-disclosure loader for SKILL.md capability packages."""
from __future__ import annotations

from pathlib import Path
import re

import yaml


class SkillLoader:
    """
    SkillLoader 用于加载和匹配 SKILL.md 能力包。

    每个 Skill 都放在 skills_dir 下的独立目录中，例如：

        skills/
        ├── coding/
        │   └── SKILL.md
        ├── research/
        │   └── SKILL.md
        └── writing/
            └── SKILL.md

    SKILL.md 通常由两部分组成：
        1. YAML Front Matter：保存 Skill 的名称和描述等元数据
        2. Markdown Body：保存 Skill 的详细使用说明

    这里采用 Progressive Disclosure（渐进式披露）策略：
        - list_skills()：只读取 metadata，不加载完整 Skill 内容
        - match()：只根据 metadata 匹配相关 Skill
        - load()：只有真正需要某个 Skill 时，才加载完整 body

    这样可以减少不必要的文件读取，同时避免把所有 Skill 的详细内容
    一次性放入 LLM 上下文中。
    """
    def __init__(self, skills_dir: str):
        """
        初始化 SkillLoader。

        参数：
            skills_dir：Skill 能力包所在的根目录。
        """

        self.skills_dir = Path(skills_dir)

    def _documents(self):
        """
        查找所有 Skill 的 SKILL.md 文件。

        glob("*/SKILL.md") 表示只查找 skills_dir 下一层目录中的
        SKILL.md，例如：

            skills/coding/SKILL.md
            skills/research/SKILL.md

        sorted() 可以保证返回顺序稳定，避免文件系统顺序不同
        导致 Skill 列表顺序发生变化。
        """
        yield from sorted(self.skills_dir.glob("*/SKILL.md"))

    @staticmethod
    def _split(text: str) -> tuple[dict, str]:
        """
        将 SKILL.md 拆分成 YAML metadata 和 Markdown body。

        一个典型的 SKILL.md 结构如下：

            ---
            name: coding
            description: Coding related capabilities
            ---

            # Coding Skill

            这里是 Skill 的详细说明。

        返回：

            metadata：YAML Front Matter 解析后的字典
            body：Skill 的 Markdown 正文
        """
        if not text.startswith("---\n"):
            return {}, text
        _, front_matter, body = text.split("---", 2)
        metadata = yaml.safe_load(front_matter) or {}
        return metadata, body.lstrip("\r\n")

    def list_skills(self) -> list[dict]:
        """
        列出所有有效 Skill 的基本信息。

        这里只读取 SKILL.md 中的 metadata，
        不读取或返回完整的 Skill Body。

        返回示例：

            [
                {
                    "name": "coding",
                    "description": "Coding related capabilities"
                },
                {
                    "name": "research",
                    "description": "Research related capabilities"
                }
            ]
        """
        skills = []
        for document in self._documents():
            metadata, _ = self._split(document.read_text(encoding="utf-8"))
            if metadata.get("name") and metadata.get("description"):
                skills.append({"name": metadata["name"], "description": metadata["description"]})
        return skills

    def load(self, name: str) -> str:
        """
        根据 Skill 名称加载完整的 Skill Body。

        例如：

            loader.load("coding")

        会找到：

            skills/coding/SKILL.md

        然后只返回 YAML Front Matter 下面的 Markdown 正文。

        注意：
            该方法是在确定需要某个 Skill 后才调用，
            体现了 Progressive Disclosure 的第二阶段。
        """

        for document in self._documents():
            metadata, body = self._split(document.read_text(encoding="utf-8"))
            if metadata.get("name") == name:
                return body
        raise KeyError(f"Unknown skill: {name}")

    def match(self, task: str) -> list[dict]:
        """
        根据任务描述匹配可能相关的 Skill。

        这里采用简单的关键词交集匹配：

            用户任务关键词
                    ↓
            Skill description 关键词
                    ↓
              求交集
                    ↓
            有共同关键词 → 匹配

        一个重要设计是：
        match() 只检查 Skill 的 metadata/description，
        不会加载完整 Skill Body。

        因此它可以先快速筛选候选 Skill，
        之后再通过 load() 显式加载真正需要的 Skill。
        """
        """Match against metadata only; bodies are loaded explicitly afterwards."""
        words = set(re.findall(r"[a-z0-9_-]+", task.lower()))
        return [skill for skill in self.list_skills()
            if words.intersection(re.findall(r"[a-z0-9_-]+", skill["description"].lower()))]