# US-072 Model Discovery Design

## Goal

Let Settings > AI discover the real models exposed by the user-configured Ollama or LM Studio endpoint, select one, persist it through the existing AI settings source of truth, and present truthful loading/error/empty states.

## Architecture decision

LM Studio remains Rex's existing OpenAI-compatible runtime path. Do not add a new Python provider name or a second provider authority. Runtime `models.llm_provider = "openai"` plus `openai.base_url` continues to drive LM Studio-compatible inference. Ollama continues to use runtime provider `ollama` plus `ollama.base_url`.

Discovery is a user-initiated Electron main-process operation. The renderer sends only a discovery kind (`ollama` or `lmstudio`); it never supplies a URL to fetch. The main process reads the canonical configured endpoint from `rex_config`, validates it as HTTP/HTTPS, applies a bounded timeout, and performs the request. This preserves the existing raw-renderer-fetch security boundary and prevents a renderer caller from turning the IPC method into an arbitrary URL fetch primitive.

## Components

### `gui/src/main/modelDiscovery.ts`

A small, dependency-free discovery module with an injectable fetch function for deterministic tests.

- Ollama: GET `<configured ollama base>/api/tags`; extract unique nonblank `models[].name` strings.
- LM Studio: GET `<configured openai base>/models`; for a normal `.../v1` base this becomes `.../v1/models`; extract unique nonblank `data[].id` strings.
- Reject missing/invalid configured endpoints before network access.
- Use `AbortController` with a 5-second timeout.
- Return structured `{ok, models, error?}` data and never return response bodies or exception payloads to the renderer.

### Settings IPC

Add a typed `rex:discoverAiModels` handler and preload/API method. The handler accepts only `ollama | lmstudio`, reads `rex_config` itself, and delegates to the discovery module.

### AI settings source of truth

Add `openaiBaseUrl` to `AiSettings` and mirror it to `openai.base_url`. A blank value clears the compatible endpoint so official OpenAI behavior remains available. Existing `models.llm_provider` remains the sole provider source of truth.

### Renderer UX

- OpenAI section gains an optional `OpenAI-compatible Base URL (LM Studio)` field.
- When that field is nonblank, show an explicit `Discover LM Studio Models` button.
- Ollama shows an explicit `Discover Models` button.
- Discovery is never automatic merely because the settings page opened.
- State is explicit: loading, network/error, successful empty list, or successful model list.
- A discovered model selection uses the existing save path (`model` for LM Studio/OpenAI-compatible; `customModelId` for Ollama), so restart/reload persistence stays within the current settings architecture.
- Existing text inputs remain available for manual IDs. Placeholders/examples are not presented as discovered availability.

## Error and security behavior

- No discovery request occurs without a user action.
- No external URL is accepted from the renderer.
- Only configured HTTP/HTTPS endpoints are used.
- Timeout/network/HTTP/schema failures produce concise provider-specific errors without raw body, stack, filesystem, credential, or exception details.
- Empty successful responses are not errors; the UI says that no models were reported by the configured endpoint.

## Testing

1. Unit-test `modelDiscovery.ts` with mocked fetch for Ollama and LM Studio success, failure, malformed/empty responses, endpoint composition, and invalid configuration.
2. Test settings build/mirror round-trip for `openaiBaseUrl` and discovered-model persistence.
3. Test the IPC handler reads configured endpoints rather than accepting renderer URLs.
4. Add renderer/source assertions for loading, error, empty, and model-selection states.
5. Run the full GUI suite, typecheck, build, lint, high-severity npm audit, the PRD Python validation set, the repository security gate, and `git diff --check`.

## Scope exclusions

No live provider calls in tests, no LM Studio Python provider class, no automatic discovery on startup, no OpenRouter discovery, and no model download/install behavior.
