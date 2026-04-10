# AskRex Assistant — Brand Assets

This directory contains official AskRex Assistant brand logo variants.

## Asset Inventory

| File | Dimensions | Use |
|------|-----------|-----|
| `icon-square.png` | 426×512 px | App icon for square icon slots (Windows taskbar, desktop shortcut) |
| `icon-circle.png` | 1024×1536 px | App icon for circular icon slots (macOS Dock, Android launcher) |
| `icon-r.png` | 1024×1536 px | Monogram / compact icon — "R" lettermark only |
| `wordmark-dark.png` | 1536×1024 px | Wordmark for use on light / white backgrounds |
| `wordmark-light.png` | 1536×1024 px | Wordmark for use on dark / coloured backgrounds |
| `wordmark-reverse.png` | 1536×1024 px | Reversed/inverted wordmark for alternate dark backgrounds |
| `primary-horizontal.png` | 1536×1024 px | Primary logo lockup — icon + wordmark, horizontal layout |
| `stacked.png` | 1536×1024 px | Stacked logo lockup — icon above wordmark |
| `favicon.ico` | 16/32/48 px | Browser tab favicon (multi-size ICO) |
| `favicon.png` | 314×376 px | Favicon source (PNG, for platforms that accept PNG favicons) |

## Usage Guidelines

- **Web / README**: use `primary-horizontal.png` or `stacked.png` at display width ≤ 400 px.
- **Electron app icon**: use `icon-square.png` (set in `gui/package.json` → `"icon"`).
- **Browser favicon**: use `favicon.ico` via `<link rel="icon">`.
- **Apple touch icon**: use `icon-square.png` via `<link rel="apple-touch-icon">`.
- **Dark-mode UIs**: use `wordmark-light.png` or `wordmark-reverse.png`.
- **Light-mode UIs**: use `wordmark-dark.png` or `primary-horizontal.png`.

## Source Files

The underscore-named originals (`icon_square.png`, `icon_circle.png`, `icon_r_only.png`,
`primary_horizontal.png`) are retained for backward compatibility. All new references
should use the hyphenated canonical names above.
