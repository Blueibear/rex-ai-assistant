"""Identity (whoami, identify, voice-id) commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
from datetime import UTC


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_whoami(args: argparse.Namespace) -> int:
    """Show the active user for this session."""
    from rex.identity import get_session_user, list_known_users, resolve_active_user

    session_user = get_session_user()
    resolved = resolve_active_user()

    if resolved:
        source = "session" if session_user == resolved else "config"
        print(f"Active user: {resolved} (source: {source})")
    else:
        print("No active user set.")
        print()
        print("Set one with:")
        print("  rex identify --user <id>     (non-interactive)")
        print("  rex identify                 (interactive selection)")

    known = list_known_users()
    if known:
        print()
        print("Known users:")
        for u in known:
            marker = " *" if u["id"] == resolved else ""
            role_str = f" ({u['role']})" if u.get("role") else ""
            print(f"  {u['id']}: {u['name']}{role_str}{marker}")

    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    """Set the active user for this session."""
    from rex.identity import list_known_users, set_session_user

    explicit = getattr(args, "user", None)

    if explicit:
        set_session_user(explicit)
        print(f"Active user set to: {explicit}")
        return 0

    # Interactive selection
    known = list_known_users()
    if not known:
        print("No known users found in Memory/ profiles.")
        print("Create a user profile directory under Memory/<user_id>/core.json first.")
        return 1

    print("Select a user:")
    for i, u in enumerate(known, 1):
        role_str = f" ({u['role']})" if u.get("role") else ""
        print(f"  {i}. {u['name']} [{u['id']}]{role_str}")

    print()
    try:
        choice = input("Enter number or user ID: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1

    if not choice:
        print("No selection made.")
        return 1

    # Try numeric selection
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(known):
            selected = known[idx]["id"]
            set_session_user(selected)
            print(f"Active user set to: {selected}")
            return 0
    except ValueError:
        pass

    # Try by user ID
    valid_ids = [u["id"] for u in known]
    if choice in valid_ids:
        set_session_user(choice)
        print(f"Active user set to: {choice}")
        return 0

    print(f"Unknown user: {choice}")
    print(f"Valid users: {', '.join(valid_ids)}")
    return 1


def cmd_voice_id(args: argparse.Namespace) -> int:
    """Manage voice identity enrollment, calibration, and status."""
    subcommand = getattr(args, "voice_id_command", None)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_voice_id_config():
        """Load voice_identity section from rex_config.json."""
        from rex.voice_identity.types import VoiceIdentityConfig

        try:
            from rex.config_manager import load_config as _load_json

            raw = _load_json()
            vi_dict = raw.get("voice_identity", {})
            return VoiceIdentityConfig(
                enabled=vi_dict.get("enabled", False),
                accept_threshold=float(vi_dict.get("accept_threshold", 0.85)),
                review_threshold=float(vi_dict.get("review_threshold", 0.65)),
                embedding_dim=int(vi_dict.get("embedding_dim", 192)),
                model_id=str(vi_dict.get("model_id", "synthetic")),
            )
        except Exception as exc:
            from rex.voice_identity.types import VoiceIdentityConfig

            print(f"Warning: Could not load voice_identity config: {exc}", flush=True)
            return VoiceIdentityConfig()

    def _get_store():
        from pathlib import Path

        from rex.voice_identity.embeddings_store import EmbeddingsStore

        memory_dir = Path(__file__).resolve().parent.parent / "Memory"
        return EmbeddingsStore(memory_dir)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    if subcommand == "status":
        vi_cfg = _load_voice_id_config()
        store = _get_store()
        enrolled = store.list_enrolled_users()
        user_filter = getattr(args, "user", None)

        print("Voice Identity Status")
        print("=" * 50)
        print(f"  Enabled          : {'yes' if vi_cfg.enabled else 'no'}")
        print(f"  Backend/model_id : {vi_cfg.model_id}")
        print(f"  Accept threshold : {vi_cfg.accept_threshold}")
        print(f"  Review threshold : {vi_cfg.review_threshold}")
        print(f"  Embedding dim    : {vi_cfg.embedding_dim}")

        if user_filter:
            emb = store.load(user_filter)
            enrolled_str = user_filter if emb is not None else f"{user_filter} (not enrolled)"
            print(f"  User filter      : {enrolled_str}")
        else:
            print(f"  Enrolled users   : {len(enrolled)}")
            if enrolled:
                for uid in enrolled:
                    emb = store.load(uid)
                    samples = emb.sample_count if emb else 0
                    updated = emb.updated_at[:10] if emb and emb.updated_at else "unknown"
                    print(f"    - {uid}  (samples={samples}, updated={updated})")

        from rex.voice_identity.optional_deps import check_voice_id_available

        real_available = check_voice_id_available()
        print(
            f"  Real backend     : {'available' if real_available else 'not installed (synthetic only)'}"
        )
        return 0

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    if subcommand == "list":
        store = _get_store()
        enrolled = store.list_enrolled_users()
        if not enrolled:
            print("No users enrolled for voice identity.")
            print("Use 'rex voice-id enroll --user <id> --wav <path>' to enroll.")
            return 0
        print(f"Enrolled users ({len(enrolled)}):")
        for uid in enrolled:
            emb = store.load(uid)
            if emb:
                print(
                    f"  {uid}  model={emb.model_id}  samples={emb.sample_count}"
                    f"  updated={emb.updated_at[:10] if emb.updated_at else 'unknown'}"
                )
            else:
                print(f"  {uid}  (could not load embedding)")
        return 0

    # ------------------------------------------------------------------
    # enroll
    # ------------------------------------------------------------------

    if subcommand == "enroll":
        import wave as _wave
        from pathlib import Path

        vi_cfg = _load_voice_id_config()

        if not vi_cfg.enabled:
            print(
                "Error: Voice identity is disabled in config. "
                "Set voice_identity.enabled=true in config/rex_config.json first."
            )
            return 1

        user_id = getattr(args, "user", None)
        wav_path = getattr(args, "wav", None)
        replace = getattr(args, "replace", False)
        yes = getattr(args, "yes", False)

        if not user_id:
            print("Error: --user is required for enroll.")
            return 1
        if not wav_path:
            print("Error: --wav is required for enroll.")
            return 1

        wav_file = Path(wav_path)
        if not wav_file.exists():
            print(f"Error: WAV file not found: {wav_path}")
            return 1
        if not wav_file.is_file():
            print(f"Error: Not a file: {wav_path}")
            return 1

        # Select embedding backend
        try:
            from rex.voice_identity.optional_deps import get_embedding_backend

            backend = get_embedding_backend(vi_cfg.model_id, dim=vi_cfg.embedding_dim)
        except ImportError as exc:
            print(f"Error: Cannot load embedding backend: {exc}")
            return 1
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        # Read WAV PCM bytes
        try:
            with _wave.open(str(wav_file), "rb") as wf:
                pcm_bytes = wf.readframes(wf.getnframes())
        except Exception as exc:
            print(f"Error: Could not read WAV file {wav_path!r}: {exc}")
            return 1

        if not pcm_bytes:
            print(f"Error: WAV file {wav_path!r} contains no audio frames.")
            return 1

        # Check existing enrollment
        store = _get_store()
        existing = store.load(user_id)

        if existing and not replace:
            print(
                f"User {user_id!r} already has an enrollment "
                f"(samples={existing.sample_count}, model={existing.model_id})."
            )
            print("Use --replace to wipe and re-enroll.")
            return 1

        if replace and existing:
            print(f"Replacing existing enrollment for {user_id!r}.")

        # Confirmation gate
        if not yes:
            print(f"About to enroll user {user_id!r} using backend {vi_cfg.model_id!r}.")
            print("Re-run with --yes to confirm.")
            return 1

        # Generate embedding
        try:
            vector = backend.embed(pcm_bytes)
        except Exception as exc:
            print(f"Error: Embedding generation failed: {exc}")
            return 1

        # Store embedding
        from datetime import datetime

        from rex.voice_identity.types import VoiceEmbedding

        label = getattr(args, "label", None) or ""
        sample_count = 1
        if existing and not replace:
            # accumulate — not reached because replace guard above, but kept for safety
            sample_count = existing.sample_count + 1

        emb = VoiceEmbedding(
            vector=vector,
            model_id=vi_cfg.model_id,
            sample_count=sample_count,
            updated_at=datetime.now(UTC).isoformat(),
        )
        try:
            store.save(user_id, emb)
        except Exception as exc:
            print(f"Error: Could not save enrollment for {user_id!r}: {exc}")
            return 1

        label_str = f" ({label})" if label else ""
        print(
            f"Enrolled {user_id!r}{label_str} using backend {vi_cfg.model_id!r}. "
            f"Embedding dim={len(vector)}."
        )
        return 0

    # ------------------------------------------------------------------
    # calibrate
    # ------------------------------------------------------------------

    if subcommand == "calibrate":
        vi_cfg = _load_voice_id_config()
        store = _get_store()
        yes = getattr(args, "yes", False)
        write_config = getattr(args, "write_config", False)

        from rex.voice_identity.calibration import calibrate as _calibrate

        report = _calibrate(store, vi_cfg)

        print("Voice Identity Calibration Report")
        print("=" * 50)
        print(f"  Enrolled users : {', '.join(report.enrolled_users) or '(none)'}")
        if report.min_inter_user_similarity is not None:
            print(f"  Min inter-user similarity : {report.min_inter_user_similarity:.4f}")
        print(f"  Recommended accept_threshold : {report.recommended_accept_threshold}")
        print(f"  Recommended review_threshold : {report.recommended_review_threshold}")
        print()
        for note in report.notes:
            print(f"  Note: {note}")

        if write_config:
            if not yes:
                print()
                print("Use --yes to confirm writing thresholds to config/rex_config.json.")
                return 1

            try:
                from rex.config_manager import load_config as _load_json
                from rex.config_manager import save_config as _save_json

                raw = _load_json()
                vi_section = raw.get("voice_identity", {})
                vi_section["accept_threshold"] = report.recommended_accept_threshold
                vi_section["review_threshold"] = report.recommended_review_threshold
                raw["voice_identity"] = vi_section
                _save_json(raw)
                print()
                print(
                    f"Config updated: accept_threshold={report.recommended_accept_threshold}, "
                    f"review_threshold={report.recommended_review_threshold}"
                )
            except Exception as exc:
                print(f"Error: Could not write config: {exc}")
                return 1

        return 0

    print("Unknown voice-id subcommand. Use 'rex voice-id --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # identity (whoami / identify)
    whoami_parser = subparsers.add_parser(
        "whoami",
        help="Show the active user for this session",
        description="Display the currently active user identity (from config, session, or --user flag).",
    )
    whoami_parser.set_defaults(func=_cli().cmd_whoami, command="whoami")

    identify_parser = subparsers.add_parser(
        "identify",
        help="Set the active user for this session",
        description=(
            "Select or set the active user identity. "
            "Used when voice/speaker recognition is unavailable or uncertain."
        ),
    )
    identify_parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Set active user non-interactively (e.g. --user cole)",
    )
    identify_parser.set_defaults(func=_cli().cmd_identify, command="identify")


def register_voice_id(subparsers: argparse._SubParsersAction) -> None:
    """Register the voice-id subcommand (kept separate to preserve help ordering)."""
    # voice-id (voice speaker identity enrollment, calibration, and status)
    voice_id_parser = subparsers.add_parser(
        "voice-id",
        help="Manage voice speaker identity (enrollment, calibration, status)",
        description=(
            "Enroll voice samples, calibrate recognition thresholds, and inspect "
            "voice identity status.  Voice identity must be enabled in "
            "config/rex_config.json (voice_identity.enabled=true) before enrolling."
        ),
    )
    voice_id_subparsers = voice_id_parser.add_subparsers(
        title="voice-id commands",
        dest="voice_id_command",
        metavar="COMMAND",
    )

    # voice-id status
    voice_id_status = voice_id_subparsers.add_parser(
        "status",
        help="Show voice identity configuration and enrollment status",
        description="Display the current voice identity configuration and enrolled users.",
    )
    voice_id_status.add_argument(
        "--user",
        type=str,
        default=None,
        help="Show enrollment status for a specific user ID",
    )
    voice_id_status.set_defaults(func=_cli().cmd_voice_id, voice_id_command="status")

    # voice-id list
    voice_id_list = voice_id_subparsers.add_parser(  # noqa: F841
        "list",
        help="List enrolled users",
        description="List all users with stored voice embeddings.",
    )
    voice_id_list.set_defaults(func=_cli().cmd_voice_id, voice_id_command="list")

    # voice-id enroll
    voice_id_enroll = voice_id_subparsers.add_parser(
        "enroll",
        help="Enroll a voice sample for a user",
        description=(
            "Read a WAV file, generate a speaker embedding, and store it for "
            "the specified user.  voice_identity.enabled must be true in config."
        ),
    )
    voice_id_enroll.add_argument(
        "--user",
        type=str,
        required=True,
        help="User ID to enroll (must match a Memory/ profile directory)",
    )
    voice_id_enroll.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Path to a WAV file containing the voice sample",
    )
    voice_id_enroll.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional human-readable label for this enrollment",
    )
    voice_id_enroll.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Replace (wipe) any existing enrollment for this user",
    )
    voice_id_enroll.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Confirm enrollment (required to execute)",
    )
    voice_id_enroll.set_defaults(func=_cli().cmd_voice_id, voice_id_command="enroll")

    # voice-id calibrate
    voice_id_calibrate = voice_id_subparsers.add_parser(
        "calibrate",
        help="Compute recommended recognition thresholds from enrolled samples",
        description=(
            "Analyse enrolled embeddings and recommend accept_threshold and "
            "review_threshold values.  Use --write-config --yes to apply them."
        ),
    )
    voice_id_calibrate.add_argument(
        "--user",
        type=str,
        default=None,
        help="Limit calibration to a specific user (currently informational only)",
    )
    voice_id_calibrate.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Confirm writing thresholds to config (required with --write-config)",
    )
    voice_id_calibrate.add_argument(
        "--write-config",
        action="store_true",
        default=False,
        dest="write_config",
        help="Write recommended thresholds to config/rex_config.json",
    )
    voice_id_calibrate.set_defaults(func=_cli().cmd_voice_id, voice_id_command="calibrate")

    voice_id_parser.set_defaults(func=_cli().cmd_voice_id, voice_id_command="status")
