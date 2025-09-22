"""Wrapper that starts the assistant's interactive loop."""

from rex.assistant import VoiceAssistant


def main() -> None:
    VoiceAssistant().run()


if __name__ == "__main__":
    main()
