# Assets directory

Runtime utilities populate this directory with generated audio such as the
wake-word acknowledgment tone (`wake_acknowledgment.wav`) and the placeholder
voice sample under `voices/`. The files are intentionally created on demand so
binary artifacts do not need to live in version control. Any legacy
acknowledgment recordings (for example `rex_wake_acknowledgment.wav`) are
deleted automatically the next time the helper runs so that outdated binaries
cannot re-enter Git history.

Custom wake-word models (`rex.onnx`) and any demo audio clips are ignored by
Git for the same reason—drop them into the repository when experimenting, but
expect a clean checkout to contain only source files.
