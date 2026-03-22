"""OpenClaw integration subpackage for Rex AI Assistant.

This package contains all modules that bridge Rex to the OpenClaw agent
engine.  It is the primary landing zone for Phase 2 migration work.

Submodules (added incrementally during migration):

- ``agent``   — ``RexAgent`` class that registers Rex with OpenClaw's agent API
- ``config``  — maps ``rex.config.Settings`` fields to OpenClaw agent config
- ``session`` — maps Rex user identity to OpenClaw session model
- ``tools``   — Rex tool adapters registered as OpenClaw skills

None of these submodules are imported here automatically; callers must
import them explicitly to avoid side-effects at package import time.
"""
