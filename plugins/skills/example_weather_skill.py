"""Example Rex skill: current weather lookup.

This file demonstrates the SKILL_METADATA pattern required for Rex to
auto-discover and register a skill from the ``plugins/skills/`` directory.

Required top-level dict:

    SKILL_METADATA = {
        "name": "<human-readable name>",
        "description": "<what the skill does>",
        "triggers": ["<pattern1>", "<pattern2>", ...],
    }

The ``handler`` entry in the registry will point to this file's path.
Rex invokes the skill by importing this module and calling ``run(transcript)``.
"""

SKILL_METADATA = {
    "name": "weather",
    "description": "Fetch the current weather for a location mentioned in the user's message.",
    "triggers": [
        r"what(?:'s| is) the weather",
        r"weather (?:in|for|at)",
        r"is it (?:raining|sunny|snowing|cloudy)",
        r"temperature (?:in|for|at)",
    ],
}


def run(transcript: str) -> str:
    """Return a weather summary for the location in *transcript*.

    This is a stub implementation.  Replace with a real weather API call.

    Args:
        transcript: The full user utterance that triggered this skill.

    Returns:
        A human-readable weather description.
    """
    return (
        "I don't have live weather access in this example skill. "
        "Configure a weather API key and replace this stub."
    )
