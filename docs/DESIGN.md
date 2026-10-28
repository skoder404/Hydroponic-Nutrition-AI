# DESIGN — Hydroponic Nutrition AI

> **Status:** Draft v0.2 · **Date:** 2026-10-20
> **Direction (from the team):** *Earthy + nature-based, built in React, with dynamic animations wherever they help, while staying clean and neat.*
> **Inputs:** `PROJECT_OVERVIEW.md` §5 (UI behaviour), §19 (no fusion), §21 (history), `PRD.md`.
> **Implementation note:** Next.js *is* React; the UI is React components throughout (see TECH_RULES). Animation uses **Framer Motion** (package name/version to be verified at setup; it may be published as `motion`) plus plain CSS for micro-interactions.
> Colour contrast ratios in this file were **computed** (WCAG formula) for the listed pairs and must be re-checked in the implementation task `T7-002`.

---

## Design Philosophy

### Visual Style: "Greenhouse journal"
Warm cream paper background, deep forest-green structure, clay/terracotta accents, soft sand panels, and hand-drawn-feeling leaf line art used sparingly. The look is **calm and organic, but the layout stays rigorous**: an 8-pt grid, generous whitespace, one primary action per view, and data (results, uncertainty, limitations) always more prominent than decoration.

### The "clean and neat" guardrails for animation
Animation is welcome, but it must follow these rules so the UI never feels busy:
1. **Purpose first.** Each animation must do one of: *explain* (progress, cause/effect), *guide* (focus, entry), *reassure* (loading, success), or *delight* (only on Home/About/empty states).
2. **Dense screens stay still.** Tables, forms with errors, and the limitations text never animate beyond a 150 ms fade.
3. **Budget:** UI feedback 120–200 ms; entrances 250–450 ms; decorative loops only on Home/About/loading and always pausable.
4. **Animate only `transform` and `opacity`** (plus SVG stroke-dashoffset) to stay smooth on laptops without a GPU.
5. **`prefers-reduced-motion` is honoured everywhere:** replace movement with a simple fade or nothing; stop all loops.
6. **Uncertainty is never "pretty-fied":** low-confidence results do not get celebratory motion.

### Branding Direction
- Name: **Hydroponic Nutrition AI**; mark = leaf with a water droplet (SVG, animated draw-in on first load only).
- Voice: plain, careful, non-alarmist. Words such as **predicted**, **likely**, **uncertain**, **advisory**; never "diagnosed" or "guaranteed".

### UX Goals
1. **Honesty over polish.** Uncertainty and limitations are first-class UI, not footnotes.
2. **Separate evidence.** Image and sensor results sit in separate panels; no merged score (Overview §19).
3. **Never fake liveness.** Every value shows its source (`manual / live / imported`) and observation time; live values show a freshness timer.
4. **Graceful failure.** Loading, empty, error, retry, and device-unavailable states exist for every screen.
5. **Low friction.** Crop, then data, then result.

---

## Information Architecture

### Navigation Structure
```
Top bar:  [Logo]  Analyze  Live  History  Models & Limits  About
Footer:   Advisory disclaimer · Dataset/licence note · Version
```

### Screen Hierarchy
```
/                     S1 Home & Crop Select
/analyze              S2 Analyze (source + inputs)  → S3 Results (same route, result state)
/live                 S8 Live Monitor (campus sensors + camera)
/history              S4 History list
/history/[id]         S5 Record detail
/models               S6 Models & Limitations
/about                S7 About / How it works
```

### User Flow (ASCII)
```
S1 select crop ──► S2 choose mode (image | sensors | both) and source (manual | upload | campus live)
                        │
        ┌───────────────┴───────────────┐
   image path                      sensor path
 upload / camera frame            enter / campus reading
   │ invalid ─► inline error        │ stale/invalid ─► explained, manual fallback
   ▼                               ▼
 segment (if required) ─ fail ─► warning + [Retry]
   ▼                               ▼
 image prediction                sensor prediction
        └───────────────┬───────────────┘
                        ▼
                 S3 results (two separate panels)
                        ▼
              recommendation (Gemini ▸ rules fallback)
                        ▼
            [Save record] ─► S4 History ─► S5 Detail
S8 Live Monitor ──► [Analyze this reading / frame] ──► S2 (prefilled, source = live)
```

---

## Motion System (what moves, and why)

| Where | Animation | Duration | Purpose | Reduced-motion fallback |
|---|---|---|---|---|
| S1 hero | Line-art plant "grows" (stem then leaves via SVG stroke draw), runs once | ~1.2 s | Delight + brand | Static final drawing |
| S1 crop cards | Staggered fade-up on load (60 ms stagger); hover lifts 3 px and a leaf icon tilts 6° | 300 ms / 150 ms | Guide + affordance | Fade only |
| Page transitions | Cross-fade + 8 px upward slide | 250 ms | Continuity | Instant |
| Step progress (S2/S3) | A **vine** grows along the step bar as *Validating, Segmenting, Classifying, Recommending* complete | per step | Explain pipeline progress | Step text + check marks only |
| Loading | Droplet fills a leaf outline (indeterminate) | loop 1.4 s | Reassure | Static "Working…" text + spinner-free progress label |
| ConfidenceMeter | Bar fills to the value with ease-out; tier icon pops once | 500 ms | Explain magnitude | Bar shown at final value |
| Sensor reading rows | Needle/bar moves into position; value counts up | 400 ms | Show reading vs band | Final value only |
| Result panels | Fade-up, image panel then sensor panel (80 ms offset) | 300 ms | Guide reading order | Fade only |
| Leaf cards | Cards stagger in (50 ms); selecting a card fades an outline around its leaf on the photo | 250 ms | Link card and leaf | Fade only |
| Mask overlay toggle | Cross-fade between original and mask overlay | 200 ms | Explain segmentation | Instant swap |
| Save success | Leaf-check icon draws in; toast slides in | 300 ms | Reassure | Static check |
| S8 live values | Gentle pulse on a value *only when new data arrives*; freshness ring depletes toward "stale" | 600 ms / continuous ring | Show liveness honestly | Numeric "updated 12 s ago" only |
| Background | Very slow drifting leaf shapes at 4–6% opacity on Home/About only | loop 40 s+ | Atmosphere | Disabled |
| Errors | Inline 150 ms fade; no shake | 150 ms | Calm | Same |

Implementation rules: a single `<MotionProvider>` reads `prefers-reduced-motion` and exposes shared variants (`fadeUp`, `stagger`, `grow`); no component defines its own durations outside the tokens; decorative loops pause when the tab is hidden.

---

## Screen Specifications

### S1: Home & Crop Select
| Aspect | Spec |
|---|---|
| **Purpose** | Orient, choose crop, set expectations (advisory, uncertainty). |
| **Components** | Animated hero (growing plant), 3 `CropCard`s (Lettuce / Spinach / Tomato with line-art, supported inputs, class counts), advisory banner, "Live Monitor" shortcut, "Models & Limits" link. |
| **Layout** | Centered container (max 1120 px); hero, 3-column card grid, banner. |
| **Interactions** | Card click/Enter → `/analyze?crop=…`. Hover lift + leaf tilt. Tomato card shows "Sensor mode: experimental". |
| **Empty state** | N/A (static). |
| **Error state** | `/health` failure → non-blocking banner "Backend unreachable — analysis unavailable" + Retry. |
| **Loading state** | Skeleton cards (≤ 1 s, then render regardless). |
| **Responsive** | ≥1024: 3 columns; 640–1023: 2+1; <640: 1 column, hero art scaled down. |

### S2: Analyze (Inputs)
| Aspect | Spec |
|---|---|
| **Purpose** | Collect image and/or sensor data with strict validation, from manual entry, upload, or the campus server. |
| **Components** | `CropSwitcher`, `ModeSelector` (Image / Sensors / Both), `SourceTabs` (Manual · Upload · Campus live), `InputModeToggle` (Leaf photo / Whole plant, always explicit), `ImageDropzone` (preview, remove, rules), `CampusCameraPicker` (grab current frame, shows capture time), `SensorForm` (fields with units and band hints), `CampusSensorCard` (latest reading with freshness), `AnalyzeButton`, `InputHelp` drawer (whole-plant vs leaf-crop). |
| **Layout** | Desktop: left = inputs, right = sticky "What will be used" summary. Mobile: single column, summary becomes an accordion. |
| **Interactions** | Drop/click to upload; client checks (type/size) plus server validation. Field blur validates. `Analyze` disabled until inputs are valid. Campus source: values prefilled but editable; stale or unavailable data shows why and offers manual entry. |
| **Empty state** | Dropzone with illustration and accepted formats; sensor fields empty (never prefilled with 0). |
| **Error state** | Per-field messages naming field, unit, range; file errors name the reason; "Campus server unreachable" with Retry and manual fallback. |
| **Loading state** | Vine step indicator: *Validating → Segmenting → Classifying → Recommending*; cancel allowed. |
| **Responsive** | Touch targets ≥ 44 px; camera capture via file input on phones. |

### S3: Results
| Aspect | Spec |
|---|---|
| **Purpose** | Show separate image and sensor predictions, uncertainty, evidence, recommendation. |
| **Components** | `ResultHeader` ("Predicted — not a diagnosis"), `ImageResultPanel` (plant photo with numbered leaf overlay; `LeafGrid` of `LeafCard`s, each with display label, original class in details and its **own** `ConfidenceMeter`; plant summary = leaf count and counts per predicted class, **no averaged score**; mask overlay toggle, preprocessing chips, source chip *upload / campus camera*), `SensorResultPanel` (class, evidence rows reading vs band, experimental badge, *rule-derived* note), `UncertaintyBadge`, `WarningList`, `RecommendationCard` (summary, evidence, checks, source chip Gemini/Rules), `NotAssessedNote`, `SaveRecordButton`, `LimitationsAccordion`. |
| **Layout** | Two panels side by side; recommendation full width; limitations accordion at the bottom. One modality only → "Not assessed: …" panel. |
| **Interactions** | Toggle mask overlay; expand evidence; copy summary; Save. **No** combined score and **no** "contradiction" alerts without a validated mapping. |
| **Empty state** | "We couldn't produce a prediction" with exact reasons. |
| **Error state** | Segmentation failure or no valid leaves → warning + **Retry** / Choose different image; leaves that fail validation are listed as *not assessed* with reasons, never silently dropped. Schema/model errors → structured message. Gemini failure → silently use rules, chip shows *Rules*. DB failure → "Not saved". |
| **Loading state** | Panel skeletons; recommendation skeleton separate so predictions appear first. |
| **Responsive** | <900 px: panels stack; meters visible without hover. |

**Confidence tiers.** Thresholds come from validation data (not fixed numbers):

| Tier | Visual | Text |
|---|---|---|
| Higher | Solid forest bar + ✔ icon | "Higher confidence (model score; not a calibrated probability)" |
| Moderate | Ochre bar + ◐ icon | "Moderate confidence — consider re-checking" |
| Uncertain | Hatched stone bar + ? icon, muted card, no celebratory motion | "Uncertain — no strong recommendation" |

### S4: History
| Aspect | Spec |
|---|---|
| **Purpose** | Browse saved records with true timestamps. |
| **Components** | `FilterBar` (crop, modality, source, dates), `HistoryTable` (timestamp+TZ, crop, modality, source chip, class, confidence tier, model version), `Pagination`. |
| **Layout** | Full-width table; filters on top. |
| **Interactions** | Row → S5; sort by observed time. *Live* chip only for records flagged live at acquisition; every row carries a "Historical" tag. Motion: 150 ms fade only. |
| **Empty state** | "No saved records yet" + CTA to Analyze. |
| **Error state** | Error card + Retry; never show stale cached rows as current. |
| **Loading state** | Table skeleton. |
| **Responsive** | <768 px: stacked cards. |

### S5: Record Detail
Purpose: reproduce what was shown and why. Components: provenance card, inputs (values or image reference), predictions, confidence, model + preprocessing/rule versions, recommendation + evidence, limitations, copy JSON. Empty: "Record not found". Error: "Original image no longer available" (metadata still shown). Loading: skeleton cards. Responsive: single column, 2-col metadata grid ≥ 768 px.

### S6: Models & Limitations
Purpose: reviewer-facing transparency. Components: tabs **Models · Datasets · Limitations**; model table (crop, family, version, status, validation macro-F1, experimental flag); drawer with metrics, confusion matrix, class support, split protocol; data-type legend (measured / adjusted / derived / synthetic / rule-generated); limitations list (Overview §28). States: empty = "No promoted models" (predictions unavailable); error = registry unreadable; loading = skeleton. Responsive: tabs become a select <640 px; tables scroll inside their own container. Statement required: *scores mean agreement with project rules/labels on a documented split*.

### S7: About / How it works
Plain-language stepper and glossary (confidence, macro-F1, rule-derived label, segmentation, uncertainty). Gentle scroll-reveal allowed. States: static with an error boundary. Responsive: single column.

### S8: Live Monitor (campus server)
| Aspect | Spec |
|---|---|
| **Purpose** | Show the latest campus sensor readings and camera frame without implying more liveness than exists. The campus system grows tomato, so tomato is the default crop here. |
| **Components** | `ConnectionStatus` (connected / unreachable / stale / invalid), `LiveSensorGrid` (each reading: value, unit, observed time, freshness ring), `CameraPanel` (latest frame or stream with capture time), `AnalyzeFromLive` buttons ("Analyze these readings", "Analyze this frame"), `LimitationNote` (tomato live mapping not validated). |
| **Layout** | Two columns: sensors left, camera right; status bar on top. |
| **Interactions** | Auto-refresh at a configurable interval; pause when the tab is hidden; "Analyze" jumps to S2 with source = live. Values older than the freshness window are shown greyed and labelled **stale**, never as current. |
| **Empty state** | "No device configured — use manual entry" with link to Analyze. |
| **Error state** | "Campus server unreachable" with last-success time and Retry; individual sensors can be `missing / disconnected / invalid / stale`. |
| **Loading state** | Skeleton grid and camera placeholder. |
| **Responsive** | <900 px: stacked; camera on top. |
> `⚠️ UNCLEAR:` the campus server's protocol and whether the camera is a snapshot URL, MJPEG, or RTSP are not known yet; the panel is designed around "latest frame + timestamp" and will be adapted once the server is probed.

---

## Design System

### Colours (computed contrast on cream `#FAF6EC` unless stated)
| Token | Hex | Usage | Contrast |
|---|---|---|---|
| `--bg` (cream) | `#FAF6EC` | Page background | — |
| `--surface` | `#FFFFFF` | Cards | — |
| `--sand` | `#EFE6D2` | Panels, chips, table stripes | — |
| `--forest` | `#2F4A33` | Primary buttons, headings, nav | 9.06:1 (white text on forest 9.78:1) |
| `--moss` | `#4C6B3C` | Secondary accent, links | 5.60:1 (white on moss 6.04:1) |
| `--sage` | `#8FA77A` | **Decorative only** (illustrations, fills) | 2.44:1, never text |
| `--clay` | `#A24E2D` | Warm accent, secondary CTA | 5.31:1 (on sand 4.62:1, white on clay 5.73:1) |
| `--bark` | `#3E3022` | Body text | 11.79:1 |
| `--stone` | `#665C4D` | Secondary text | 6.08:1 (on sand 5.28:1) |

| `--border` | `#D9CDB1` | Decorative dividers | 1.46:1, decorative only |
| `--border-strong` | `#857A67` | Input/focus-adjacent borders | 3.91:1 (UI component ≥ 3:1) |
| `--water` | `#2F6F7E` | Sensor/water accents | 5.26:1 |
| `--success` | `#2E6B3A` | Success | 5.93:1 |
| `--warning` | `#8A5A00` | Moderate / caution | 5.49:1 |
| `--error` | `#A32A22` | Errors | 6.68:1 (white on error 7.21:1) |
| `--info` | `#2A5F8F` | Info banners | 6.21:1 |

Rule: **status is never conveyed by colour alone**; pair with icon and text.

### Typography
- Headings: **Fraunces** (soft serif, gives the "journal" feel); fallback `Georgia, serif`. Body/UI: **Inter**; fallback `system-ui, sans-serif`. Mono: `JetBrains Mono` / `ui-monospace`. `⚠️` confirm font licences and self-host with `next/font`.
- Scale (rem): 0.75 · 0.875 · 1 · 1.125 · 1.25 · 1.5 · 1.875 · 2.25 · 3. Weights 400/500/600/700. Body line-height 1.55; headings 1.2.

### Spacing, Grid, Shape
- Base unit 4 px (use multiples of 8 for layout): 4·8·12·16·24·32·48·64·96.
- 12-column grid, gutters 24 px (≥1024) / 16 px (<1024). Breakpoints: `sm 640`, `md 768`, `lg 1024`, `xl 1280`.
- Radius: 10 px inputs/buttons, 16 px cards, 999 px chips. Shadow: `0 1px 2px rgba(62,48,34,.08), 0 6px 16px rgba(62,48,34,.08)` (warm, low-contrast).
- Texture: optional paper-grain overlay at ≤ 3% opacity on the page background (static PNG/SVG, no animation).

### Component Library
| Component | Variants | States |
|---|---|---|
| **Button** | primary (forest), secondary (clay outline), ghost, destructive | default, hover (darken 1 step), focus-visible (2 px `--moss` outline + 2 px offset), active, disabled, loading |
| **Input** | text, number-with-unit, select, file | default, focus, filled, error (message + icon), disabled, **missing** (explicitly empty) |
| **Card** | default, result, warning, muted-uncertain | default, interactive hover, skeleton |
| **Table** | dense, striped (sand), stacked-mobile | skeleton, empty, error |
| **Modal / Drawer** | confirm, details | open/close, focus trap, Esc |
| **Toast** | success, info, warning, error | auto-dismiss 5 s (errors persist) |
| **Badge/Chip** | source (manual/upload/live/imported/mock), model (Gemini/Rules), experimental, historical, stale | static |
| **ConfidenceMeter** | higher, moderate, uncertain | animated fill; static under reduced motion |
| **LeafCard** | default, selected, uncertain, not-assessed | selected outline fade; muted when uncertain |
| **Vine progress** | 4-step | pending, active (growing), done, failed |
| **FreshnessRing** | fresh, ageing, stale | continuous (slow) / static text under reduced motion |

---

## Accessibility Requirements
- **Level:** WCAG 2.1 AA target.
- **Keyboard:** everything reachable; visible focus; dropzone has a button alternative; modal focus trap; tabs use arrow keys.
- **Screen reader:** landmarks; labelled inputs; `aria-live="polite"` for progress, toasts and *new live data*; tables with headers; meaningful alt text for illustrations (decorative art is `aria-hidden`).
- **Contrast:** values above were computed; verify every pair in `T7-002`.
- **Motion:** `prefers-reduced-motion` supported globally; no content depends on animation to be understood; no flashing.

## Micro-interactions (summary)
Hover lift (150 ms), focus ring, button press (scale .98, 80 ms), toast slide, leaf-check on save, vine growth on progress, meter fill, live-data pulse. Full table: **Motion System** above.

## Mobile Responsiveness Strategy
- **Breakpoints:** 640 / 768 / 1024 / 1280.
- **Layout shifts:** two-column → stacked; sticky summary → accordion; tables → cards; camera panel on top in S8.
- **Touch targets:** ≥ 44×44 px; `<input type="file" accept="image/*" capture>` for phone capture.
- **Performance:** lazy-load animation-heavy components and the hero art; disable decorative loops on small screens and when the tab is hidden.
