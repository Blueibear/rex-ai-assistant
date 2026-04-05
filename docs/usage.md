# AskRex Assistant — Usage Guide

## 1. Text-Based Chat Mode

Start an interactive text-based conversation with Rex (no microphone required):

```bash
python -m rex
# or
python rex_assistant.py
```

Type your messages and press Enter. Type `exit` or `quit` to stop.

**Example session:**
```
🎤 Rex assistant ready. Type 'exit' or 'quit' to stop.
You: What is the capital of France?
Rex: The capital of France is Paris.
You: exit
```

## 2. Voice Assistant Mode

Start the full voice assistant with wake word detection:

```bash
python rex_loop.py

# Override user profile
python rex_loop.py --user james

# Enable specific plugins only
python rex_loop.py --enable-plugin web_search
```

**How to use:**
1. Wait for Rex to initialize (models may take 10-30 seconds to load on first run)
2. Say your wake word (default: "rex" or "hey jarvis")
3. Wait for the acknowledgment sound
4. Speak your command within 5 seconds
5. Rex will transcribe, process, and respond with speech
6. Press `Ctrl+C` to exit

## 3. GUI Settings Editor

Rex includes a user-friendly graphical settings editor that lets you configure all environment variables without manually editing `.env` files.

**Launch the GUI:**

```bash
python gui.py
```

**Features:**

- **Dashboard Tab**: Monitor Rex assistant status and recent conversation history
- **Settings Tab**: Visual editor for ALL environment variables
  - Organized by section (Core Settings, Wakeword, Audio, LLM, TTS, etc.)
  - Smart controls: dropdowns for enums, checkboxes for booleans, spinboxes for numbers, path pickers for files
  - Help tooltips: Hover or click the "?" icon next to any setting for detailed explanations
  - Secret masking: API keys and tokens are hidden by default with Show/Hide toggles
  - Restart indicators: Settings that require restart are marked with ⚠ icon
  - Advanced section: Edit custom environment variables not in `.env.example`
  - Add custom keys: Create new environment variables on the fly

**Backup & Restore:**

The Settings editor automatically creates timestamped backups before saving. Use the Backup and Restore buttons to manage your configurations:

- **Save**: Updates `.env` with new values (creates backup first)
- **Reset to Defaults**: Restores all settings to `.env.example` defaults
- **Backup**: Manually create a backup of current `.env`
- **Restore**: Choose from previous backups to restore
- **Open .env in Notepad** (Windows): Power-user option to edit `.env` directly

All backups are stored in the `backups/` directory with timestamps like `.env.backup.20240615_143022`.

**Model Selection:**

The LLM Model field adapts based on your selected provider:
- **Transformers**: Text entry with Browse button for local model paths
- **OpenAI**: Dropdown with common models (gpt-3.5-turbo, gpt-4, etc.)
- **Ollama**: Refresh button to query local Ollama instance for installed models

**Ollama Integration:**

If you're using Ollama as your LLM provider, click the "Refresh" button next to the model dropdown to automatically populate it with models from your local Ollama installation.

## 4. Audio Device Configuration

List and configure audio devices:

```bash
# List all available audio devices
python audio_config.py --list

# Set microphone (input device)
python audio_config.py --set-input 1

# Set speakers (output device)
python audio_config.py --set-output 2

# Show current configured devices
python audio_config.py --show
```

## 5. TTS API Server

Run the standalone text-to-speech HTTP API:

```bash
python rex_speak_api.py
```

**API endpoint:** `POST http://localhost:5000/speak`

**Request:**
```json
{
  "text": "Hello, world!",
  "user": "james"
}
```

**Headers:**
```
Content-Type: application/json
X-API-Key: your-secret-key
```

**Response:** WAV audio file

**Example using curl:**
```bash
curl -X POST http://localhost:5000/speak \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"text": "Hello from Rex!", "user": "default"}' \
  --output speech.wav
```

## 5. Tool Registry & Credential Management

View registered tools and their status:

```bash
# List all tools with status
rex tools

# Verbose output with credential and health details
rex tools -v
```

Rex includes a centralized credential vault and tool registry. See:
- [docs/credentials.md](credentials.md) - Credential management
- [docs/tools.md](tools.md) - Tool registry and health checks

## 6. GitHub Integration

Rex can interact with GitHub repositories, issues, and pull requests.

**Requires:** Store your GitHub personal access token in the credential manager:
```bash
# Token is read from the credential manager under the name "github"
# Set GITHUB_TOKEN in your .env or configure via rex credential commands
```

**List repositories:**
```bash
rex gh repos
rex gh repos --type owner   # only repos you own
```

**List pull requests:**
```bash
rex gh prs owner/repo
rex gh prs owner/repo --state closed
```

**Create an issue:**
```bash
rex gh issue-create owner/repo --title "Bug: something broke" --body "Steps to reproduce..."
rex gh issue-create owner/repo --title "Feature request" --body "..." --labels bug,enhancement
```

**Create a pull request:**
```bash
rex gh pr-create owner/repo --head feature-branch --base main --title "Add feature" --body "..."
```

See [docs/github.md](github.md) for full API reference and credential setup.

## 7. Health Check & Diagnostics

Run the doctor script to verify your setup:

```bash
python scripts/doctor.py
```

**Output example:**
```
Rex Doctor
Platform: Darwin 22.6.0 (arm64)

[PASS] Python version: 3.11 detected
[PASS] ffmpeg: ffmpeg available at /opt/homebrew/bin/ffmpeg
[PASS] CUDA: CUDA not detected – running in CPU mode
[WARN] REX_SPEAK_API_KEY: not set (Required for /speak TTS endpoint)
[PASS] OPENAI_API_KEY: set
[PASS] Rate limiter storage: using in-memory backend

Summary: 1 warning(s) detected.
```

## 8. Testing Individual Components

**Test Whisper transcription:**
```bash
python scripts/manual_whisper_demo.py path/to/audio.wav --model base
```

**Test web search:**
```bash
python scripts/manual_search_demo.py "Python programming tutorials"
```

**Record custom wake word:**
```bash
python scripts/record_wakeword.py
```

## 9. Autonomous Workflows

Rex can autonomously plan and execute multi-step workflows from natural language goals.

**Plan a workflow:**
```bash
# Generate a plan
rex plan "send monthly newsletter"

# Plan and execute immediately
rex plan "check weather in Dallas" --execute
```

**Resume blocked workflows:**
```bash
# View pending approvals
rex approvals

# Approve a request
rex approvals --approve <approval_id>

# Resume execution
rex executor resume <workflow_id>
```

**Configure autonomy modes** in `config/autonomy.json`:
- **AUTO**: Low-risk operations execute automatically (weather, time, web search)
- **SUGGEST**: Medium-risk operations require confirmation (email, calendar, home control)
- **OFF**: High-risk operations must be triggered manually (OS commands, file operations)

**See [docs/autonomy.md](autonomy.md) for complete documentation.**
