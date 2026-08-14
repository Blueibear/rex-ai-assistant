# Mobile API Gateway Planning Package

This directory is the implementation and security handoff for the AskRex mobile gateway.

Read in this order:

1. [MOBILE_API_MASTER_SPEC.md](MOBILE_API_MASTER_SPEC.md) — canonical wire and security contract.
2. [MOBILE_API_THREAT_MODEL.md](MOBILE_API_THREAT_MODEL.md) — current external/mobile trust boundaries and the gated `askrex.app` threat model.
3. [MOBILE_CLIENT_CONTRACT_AUDIT.md](MOBILE_CLIENT_CONTRACT_AUDIT.md) — cross-repository findings and resolved conflicts.
4. [MOBILE_API_ARCHITECTURE.md](MOBILE_API_ARCHITECTURE.md) — component, data, identity, idempotency, voice, and deployment architecture.
5. [MOBILE_API_IMPLEMENTATION_PLAN.md](MOBILE_API_IMPLEMENTATION_PLAN.md) — implementation history and workstream plan.
6. [MOBILE_API_TEST_MATRIX.md](MOBILE_API_TEST_MATRIX.md) — required automated and real-device validation.

The threat model reflects the current S5-S8 desktop/server security boundary. It does **not** claim that the future `askrex.app` public ingress is deployed or production-ready; US-088 gates that separately.
