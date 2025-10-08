"""Backward compatibility facade for wake-word helpers.

This file exists only to support legacy imports of `wakeword_utils`.
New code should import from `rex.wakeword.utils`.
"""

from rex.wakeword.utils import *  # noqa: F401,F403
