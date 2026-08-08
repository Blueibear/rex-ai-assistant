# Wake-word reliability fixture

These WAV files are controlled synthetic speech fixtures for US-046. They contain no human recordings. They were generated locally with the Windows `System.Speech`/SAPI synthesizer using the built-in Microsoft David and Microsoft Zira English voices at the rates recorded in `manifest.json`.

The fixture intentionally includes both `hey jarvis` positives and conversational/adversarial negatives, including `jarvis` alone. It is a small controlled regression fixture, not a statistically representative deployment corpus. The tracked reliability report must therefore keep wake-word classified as beta unless broader evidence also supports promotion.
