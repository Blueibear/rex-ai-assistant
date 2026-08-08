# Wake-word Reliability Report

This report is generated deterministically by `tests/test_wakeword_reliability.py` from the tracked synthetic acoustic fixture in `tests/fixtures/wakeword/`.

## Controlled result

- Active model: built-in openWakeWord `hey jarvis`
- Detector package: `openwakeword 0.6.0`
- Model activation threshold: `0.50`
- Promotion threshold: precision >= `0.90` and recall >= `0.90`
- Confusion matrix: TP `4`, FN `0`, FP `1`, TN `7`
- Precision: **0.800**
- Recall: **1.000**
- Median positive detection latency: **1000 ms** (end of the first 1-second Rex detection window that contains an accepted native openWakeWord frame)
- Threshold result: **FAIL**
- Product classification: **beta**

The fixture is intentionally small and synthetic. Passing it would be necessary but not sufficient for a production wake-word claim; broader microphones, speakers, distances, accents, room noise, and continuous negative audio still require deployment evidence.

## Samples

| File | Expected | Phrase | Detected | First detection window end |
|---|---|---|---:|---:|
| `pos_01.wav` | positive | `hey jarvis` | yes | 1000 ms |
| `pos_02.wav` | positive | `hey jarvis` | yes | 1000 ms |
| `pos_03.wav` | positive | `hey jarvis` | yes | 1000 ms |
| `pos_04.wav` | positive | `hey jarvis` | yes | 1000 ms |
| `neg_05.wav` | negative | `hello there` | no | ? |
| `neg_06.wav` | negative | `hey james` | no | ? |
| `neg_07.wav` | negative | `where are you` | no | ? |
| `neg_08.wav` | negative | `play some music` | no | ? |
| `neg_09.wav` | negative | `good morning` | no | ? |
| `neg_10.wav` | negative | `tell me the weather` | no | ? |
| `neg_11.wav` | negative | `hey siri` | no | ? |
| `neg_12.wav` | negative | `jarvis` | yes | 1000 ms |

## Decision

The controlled threshold fails, so wake-word remains **beta** and is not part of the release-verified voice contract.
