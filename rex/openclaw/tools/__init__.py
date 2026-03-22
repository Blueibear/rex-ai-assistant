"""rex.openclaw.tools — Rex tools registered with OpenClaw's tool system.

Each submodule exposes a Python callable implementing a Rex tool and a
``register()`` function that registers it with OpenClaw when the package is
available.  When OpenClaw is not installed every ``register()`` call is a
documented no-op.

Submodules
----------
- :mod:`rex.openclaw.tools.time_tool` — ``time_now`` (current local time for a location)
- :mod:`rex.openclaw.tools.weather_tool` — ``weather`` (current weather for a location)
"""
