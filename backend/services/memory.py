"""
memory.py
---------
Manages conversation history for each chat session.

Design: simple in-memory store (dict keyed by session_id).
For a single-user MVP this is fine. For multi-user production,
you'd persist this to Redis or a database.

History format (OpenAI-style, works with LangChain):
  [
    {"role": "user",      "content": "Who is the author?"},
    {"role": "assistant", "content": "The author is..."},
    ...
  ]

We keep a sliding window of the last N turns to avoid bloating
the LLM context window on long conversations.
"""

from collections import defaultdict

# In-memory store: { session_id: [{"role": ..., "content": ...}, ...] }
_sessions: dict[str, list[dict]] = defaultdict(list)

# Keep last 6 turns (3 user + 3 assistant) — ~1500 tokens of history max
MAX_TURNS = 6


def get_history(session_id: str) -> list[dict]:
    """Return the conversation history for a session."""
    return _sessions[session_id]


def add_turn(session_id: str, question: str, answer: str) -> None:
    """
    Append a user/assistant turn to the session history.
    Trims to MAX_TURNS pairs if exceeded.
    """
    history = _sessions[session_id]
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    # Sliding window — drop oldest turns first (always in pairs)
    if len(history) > MAX_TURNS * 2:
        _sessions[session_id] = history[-(MAX_TURNS * 2):]


def clear_history(session_id: str) -> None:
    """Clear history for a session. Called when a new PDF is uploaded."""
    _sessions[session_id] = []


def format_history_for_prompt(history: list[dict]) -> str:
    """
    Format history as a readable string for injection into prompts.

    Output example:
      User: Who is the author?
      Assistant: The author is John Smith.
      User: What did he study?
      Assistant: He studied computer science at MIT.
    """
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)
