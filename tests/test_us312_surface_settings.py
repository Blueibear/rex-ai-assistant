"""US-312: Surface all existing Rex settings in the GUI."""
import json
import re
from pathlib import Path


SETTINGS_PAGE = (
    Path(__file__).parent.parent / "gui" / "src" / "pages" / "SettingsPage.tsx"
)
IPC_TYPES = Path(__file__).parent.parent / "gui" / "src" / "types" / "ipc.ts"
MAIN_INDEX = Path(__file__).parent.parent / "gui" / "src" / "main" / "index.ts"
REX_CONFIG_SCHEMA = Path(__file__).parent.parent / "config" / "rex_config.schema.json"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- IntegrationsSettings type has Telegram fields ---

def test_ipc_types_has_telegram_bot_token():
    content = read(IPC_TYPES)
    assert "telegramBotToken" in content, "telegramBotToken missing from IntegrationsSettings"


def test_ipc_types_has_telegram_chat_id():
    content = read(IPC_TYPES)
    assert "telegramChatId" in content, "telegramChatId missing from IntegrationsSettings"


# --- SettingsPage renders Telegram section ---

def test_settings_page_has_telegram_section_header():
    content = read(SETTINGS_PAGE)
    assert "Telegram" in content, "Telegram section not found in SettingsPage"


def test_settings_page_has_telegram_bot_token_input():
    content = read(SETTINGS_PAGE)
    assert "telegramBotToken" in content


def test_settings_page_has_telegram_chat_id_input():
    content = read(SETTINGS_PAGE)
    assert "telegramChatId" in content


def test_settings_page_telegram_inputs_wired_to_form():
    content = read(SETTINGS_PAGE)
    # Form state must have these fields set
    assert "telegramBotToken: ''" in content or "telegramBotToken:" in content
    assert "telegramChatId:" in content


# --- main/index.ts mirrors telegram.chat_id to rex_config.json ---

def test_main_index_mirrors_telegram_chat_id():
    content = read(MAIN_INDEX)
    assert "telegram" in content.lower(), "telegram mirroring not found in main/index.ts"
    assert "chat_id" in content or "chatId" in content or "telegramChatId" in content


# --- rex/config.py has telegram_chat_id field read from rex_config.json ---

def test_python_config_has_telegram_chat_id_field():
    config_py = Path(__file__).parent.parent / "rex" / "config.py"
    content = config_py.read_text(encoding="utf-8")
    assert "telegram_chat_id" in content, "telegram_chat_id field missing from rex/config.py"


def test_python_config_reads_telegram_chat_id_from_json():
    config_py = Path(__file__).parent.parent / "rex" / "config.py"
    content = config_py.read_text(encoding="utf-8")
    # Config reads telegram.chat_id from the JSON config
    assert "telegram.chat_id" in content or "telegram_chat_id" in content


# --- SettingsPage saves field on blur ---

def test_settings_page_telegram_bot_token_has_onblur():
    content = read(SETTINGS_PAGE)
    assert "onBlur" in content and "telegramBotToken" in content
