import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversationTurn:
    user_message: str
    assistant_answer: str
    sources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "user_message": self.user_message,
            "assistant_answer": self.assistant_answer,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }


class ConversationMemory:
    """
    Manages session-level conversational memory with sliding-window turn buffer.
    """

    def __init__(self, session_id: str = "default", max_turns: int = 5):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add_turn(
        self,
        user_message: str,
        assistant_answer: str,
        sources: Optional[List[str]] = None,
    ):
        turn = ConversationTurn(
            user_message=user_message.strip(),
            assistant_answer=assistant_answer.strip(),
            sources=sources or [],
        )
        self.turns.append(turn)

        # Enforce sliding window limit
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_history_messages(self) -> List[dict]:
        """
        Returns messages in OpenAI / Chat format: [{'role': 'user', ...}, {'role': 'assistant', ...}]
        """
        messages = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_answer})
        return messages

    def get_history_text(self) -> str:
        """
        Formats conversation turns as a readable dialogue string for the Query Condenser.
        """
        if not self.turns:
            return ""

        formatted_turns = []
        for i, turn in enumerate(self.turns, start=1):
            formatted_turns.append(f"User: {turn.user_message}\nAssistant: {turn.assistant_answer}")
        return "\n\n".join(formatted_turns)

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self):
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)
