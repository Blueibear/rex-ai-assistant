# Capability parity inventory

> **Release-truth inventory for US-064.** This document maps current AskRex capability authorities to the primary Electron product surface. Backend code, a CLI route, or an OpenClaw tool does not by itself make a capability production-ready. Runtime evidence states remain defined by `INTEGRATIONS_STATUS.md`.

## Status vocabulary

- **visible**: present in the primary Electron UI, but may still require configuration or live verification.
- **configurable**: Electron exposes the settings needed to configure the supported path.
- **disabled with explanation**: Electron may show the capability, but the unavailable action must be disabled and explain what is missing.
- **developer-only**: intentionally available through CLI/backend/operator surfaces, not the primary packaged UI.
- **missing**: backend/docs capability exists but the intended Electron surface is not yet present or truthful.

Enabled state is evidence-based: `configured` means required values are stored; it does not imply `authenticated`, `write_capable`, or `verified`. Risk uses the current product convention: **low** for read-only/non-consequential work, **sensitive** for user-data or external mutations requiring permission/confirmation, and **prohibited** where policy denies the action.

## Capability parity matrix

| Capability | Current source / authority | Electron GUI status | Enabled-state evidence | Required permissions | Health evidence | Operation | Risk | Verification support | Gap owner |
|---|---|---|---|---|---|---|---|---|---|
| Text chat | `rex.assistant`, chat bridges, Electron chat handlers | visible | session identity + available model path | authenticated local user/session | reply/error event | read + generated response | low | response delivered to requesting session | US-094 to US-097 converge runtime contract |
| Voice interaction | `rex.voice_loop`, voice bridges, Electron voice handlers | visible | microphone/TTS/wake configuration | microphone + explicit user identity | structured voice/runtime state | read + local audio output | sensitive | partial automated evidence; physical mic/speaker remains external | US-068 to US-070, US-074, US-079, US-100, US-102, US-103 |
| Home Assistant | Electron `integrationInventory.ts`, `rex.home_assistant`, OpenClaw HA tool | configurable | saved base URL/token plus live auth state | authenticated HA credentials; confirmation for sensitive writes | canonical integration state + mutation result | read + mutate | sensitive | writes distinguish verified / attempted-unverified / denied / failed | US-080 improves UX; existing verification contract retained |
| Email | Electron integration settings, IMAP/SMTP backend, OpenClaw email tool | visible; send disabled with explanation | account/server credentials | account credentials; send permission for external mutation | configured/auth evidence where supported | read + draft; backend send path exists | sensitive | GUI does not claim send success; draft/copy only | US-065 configure truth; US-081 Outlook end-to-end |
| Outlook / Microsoft Graph | docs/provider code references | missing | unavailable until Graph OAuth is implemented and verified | Microsoft OAuth scopes | none production-grade today | read + mutate target | sensitive | unavailable; must not claim connected | US-081 |
| Calendar / ICS | Electron integration settings + backend calendar provider | configurable | ICS/provider configuration | calendar read credentials | configured/read evidence | read | sensitive | reads can be evidenced; provider writes unavailable | US-065; US-081 for Outlook path |
| SMS (Twilio) | backend/direct route + OpenClaw `sms_tool.py` | developer-only in primary navigation | complete Twilio credentials + tested write capability | Twilio credentials + external-send permission | integration state; delivery remains external | mutate | sensitive | no delivery claim without provider evidence | US-082; primary navigation intentionally remains absent |
| Phone (Twilio) | backend integration inventory | configurable/experimental settings, not a primary workflow | complete Twilio credentials | Twilio credentials + call permission | configuration/test evidence only | mutate | sensitive | live calling externally verified | US-065 / US-082 policy truth |
| Web search | search provider config + SerpAPI/Brave credential references | visible in Integrations; configuration path is incomplete/misleading | stored provider key only proves configured | provider API credential | no general connected proof from stored key | read | low | result freshness/provider failures must be surfaced | US-065 configure truth; US-077 current-info routing |
| OpenAI | Electron AI settings + provider client | configurable | stored API key | provider credential | live request required | read/generate | low | provider response/error | US-071 provider persistence; US-110/US-111 routing/reliability |
| OpenRouter | provider/runtime config | missing or non-parity in current Electron inventory | stored key/config if present | provider credential | live request required | read/generate | low | provider response/error | US-071, US-110, US-111 |
| Ollama | Electron AI settings + runtime provider | configurable | configured base URL/model | local endpoint access | endpoint/model discovery required | read/generate | low | live local request | US-072 discovery; US-099 warm runtime |
| LM Studio | runtime/provider support | missing configuration parity | configured endpoint/model if manually supplied | local endpoint access | endpoint/model discovery required | read/generate | low | live local request | US-072 |
| OpenClaw gateway | `rex.openclaw`, Electron integration inventory | configurable, experimental | gateway URL + vault token; disabled by default | gateway credential + Rex permission policy | GUI may prove reachable; auth/tool capability remain separate | read + mutate depending tool | sensitive | Rex remains authority; each consequential tool result must be verified | US-113 dynamic sync; US-114 reconnect/verification |
| OpenClaw registered tools | `rex/openclaw/tool_registry.py`, `rex/openclaw/tools/*` | developer-only unless separately surfaced | registry metadata + provider configuration | per-tool policy/credentials | registry health checks vary by tool | read + mutate | low/sensitive by tool | tool-specific verification; metadata never widens authority | US-106 canonical registry; US-107 permission retrieval; US-113 sync |
| Shopping list | shopping bridge/backend user data | visible capability path is incomplete across chat/voice | scoped user/session data | authenticated user; household rules where applicable | local persistence result | read + mutate | sensitive | verify persisted list mutation | US-084 |
| Memory | memory APIs/bridges and scoped stores | visible in parts; controls incomplete | explicit user/household scope | authenticated identity + memory scope | persistence/retrieval evidence | read + mutate | sensitive | scoped write/readback where applicable | US-085; US-087 identity unification; US-105/US-112 caching/experience memory |
| Profiles / users | profile store + Electron Users/Profile surfaces | visible; identity semantics not fully unified | explicit session/profile identity | authenticated user; admin/owner for cross-user changes | profile read/write result | read + mutate | sensitive | fail-closed identity and scoped persistence | US-066 avatar/settings split; US-087 unification |
| Chat history | history bridges/store | missing selectable per-user product surface | authenticated user scope | user identity | persisted conversation records | read + mutate/delete | sensitive | scoped retrieval/deletion evidence | US-083; US-087 |
| Document upload / indexing | file extract/vector backend paths | missing | none production-ready until scoped indexing is exposed | user identity + file permission + memory/index scope | parse/index result | read + mutate index | sensitive | index success must be scoped and queryable | US-086 |
| Telegram | Electron integration inventory | configurable/experimental | chat ID + vault bot token | Telegram bot credential | limited configuration evidence | mutate/notify | sensitive | delivery externally verified | US-065 configure truth |
| MQTT | Electron inventory placeholder | disabled with explanation / unimplemented configuration | currently false in Electron inventory | broker credential if implemented | unavailable today | read + mutate target | sensitive | none production-grade | US-065 |
| Push notifications | Electron inventory placeholder | disabled with explanation / unimplemented configuration | currently false in Electron inventory | notification provider/device permission | unavailable today | mutate/notify | sensitive | none production-grade | US-065; mobile notification work remains outside current capability claim |
| Browser automation | CLI/OpenClaw/operator tooling | developer-only | environment/tool availability | explicit desktop/browser permission | tool execution evidence | read + mutate | sensitive | action-specific verification | US-106/US-109 registry/action lifecycle |
| Windows / desktop control | CLI/operator tools and Desktop Commander style adapters | developer-only | local environment availability | explicit desktop permission | tool execution evidence | mutate | sensitive | action-specific verification | US-106/US-109 |
| WordPress | OpenClaw WordPress tool | developer-only | site credentials/config | site credential + permission | health/read result | mostly read | sensitive | live site remains externally verified | US-106/US-113 |
| WooCommerce | OpenClaw WooCommerce tool | developer-only | store credentials/config | store credential + approval for writes | health/read result | read + mutate | sensitive | approval-gated writes require result verification | US-106/US-109/US-113 |
| Plex | OpenClaw Plex tool | developer-only | server/token config | Plex credential | live service health | read + mutate depending action | sensitive | hardware/service-dependent | US-106/US-113 |
| Time / weather | OpenClaw/local tools | developer-only as direct tool cards; available through assistant routing | tool/provider availability | normally none or provider credential | tool response | read | low | response/error returned | US-106/US-107 |
| Mobile API gateway | `rex.mobile_api` and mobile capability modules | developer-only backend; companion pre-release | JWT secret + paired device/session/grants | paired device + least-privilege scopes | truthful mobile capability endpoint | read + mutate only where implemented | sensitive | capability false for scaffolded endpoints; physical device still external | US-088; US-087 identity semantics |

## Migration appendix: current registries to canonical Capability Registry

US-106 completed the runtime metadata consolidation. The canonical `Capability` / Tool Card authority now lives in `rex/capabilities/registry.py`; executable and OpenClaw registries are compatibility adapters over that authority. US-064 remains the release-truth inventory for UI parity.

| Current registry / metadata source | Current authority and consumers | Duplicate or divergent metadata | Target adapter into canonical Capability Registry |
|---|---|---|---|
| `rex/capabilities/registry.py` | **Canonical Capability / Tool Card metadata authority** consumed by capability query/runtime code and executable adapters | legacy capability fields remain as compatibility aliases | owns stable ID/source/schema/enabled/permissions/health/operation/risk/verification/description/examples and rejects divergent duplicate metadata |
| `rex/tools/registry.py` | executable local tool handlers and selection compatibility | previously duplicated description/config/security metadata | binds handlers to canonical Tool Cards; the default singleton shares the global Capability Registry and rechecks live user permissions |
| `rex/openclaw/tool_registry.py` | OpenClaw metadata, health checks, planner compatibility | remote metadata previously mirrored by replacement into the executable registry | adapts remote cards through the canonical registry; local security metadata cannot be overwritten and unknown remote tools enter conservatively |
| `gui/src/main/integrationInventory.ts` | Electron integration/capability cards and settings/status consumers | names, configure routes, enabled state and health presentation are Electron-specific duplicates | Electron presentation adapter derived from canonical capability records plus local settings routes |
| `rex/integration_state.py` and provider status helpers | canonical evidence vocabulary for integration readiness | state is sometimes re-expressed by GUI/tool-specific booleans | health/evidence adapter; retain evidence vocabulary as authoritative state semantics |
| mobile capability/grant modules under `rex/mobile_*` | paired-device capabilities and least-privilege grants | mobile surface has a separate capability shape and false scaffolding flags | mobile adapter filters canonical records by device grants and implemented endpoint support |
| CLI command registrations under `rex/cli_commands/` | command discovery and operator/developer entry points | command availability can be mistaken for product capability/readiness | CLI adapter maps commands to capability operations without changing GUI tier |
| `SURFACE-CLASSIFICATION.md` and `INTEGRATIONS_STATUS.md` | release classification and integration support contracts | documentation can drift from runtime registries | generated/validated documentation view sourced from canonical registry plus release-evidence annotations |

### Canonical record fields required by US-106

The future registry must be able to express, at minimum: stable capability ID/name, source/provider, product tier/surface visibility, enabled/configured state, required permissions/scopes, health/evidence state, supported operations (read/mutate), risk tier, verification contract, configure/status route where applicable, and adapter/provider-specific metadata. A consumer may narrow a capability based on identity, permissions, device grants, or health; it must never widen authority from metadata alone.

## Production-facing conclusions

1. Electron is the primary shippable user surface, but it is not yet at parity with every backend/CLI/OpenClaw capability.
2. OpenClaw is optional and experimental. Its registry supplies capability candidates, not permission or verification authority.
3. Stored credentials mean only `configured`; production-ready claims require an appropriate GUI/status surface or an explicit `developer-only` classification.
4. SMS remains callable through backend/direct routes where configured and permitted, but it is intentionally absent from primary navigation.
5. Mobile remains paired, least-privilege, and pre-release; false capability flags for unimplemented endpoints are correct behavior, not missing production claims.
