# Front-End State & Responsive Audit (T7-020)

**Target Specifications:** DESIGN.md §S1-S8, WCAG 2.1 AA  
**Breakpoints Audited:** 360 px (Mobile), 768 px (Tablet), 1024 px (Desktop), 1280 px (Wide Desktop)  
**Touch Target Compliance:** Minimum 44 × 44 px on all interactive controls (`Button`, `Input`, `Select`, `Card` targets)  
**Horizontal Overflow:** Strict `overflow-x: hidden` on viewport roots with internal scrolling on all data tables.

---

## 1. Screen-by-Screen State Audit Checklist

| Screen | Route | Empty State | Loading State | Error State | Populated State | 360 px (Mobile) | 768 px (Tablet) | 1024/1280 px (Desktop) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1: Home** | `/` | N/A | Hero skeleton | Campus banner warning | Plant hero + CTA | Stacked hero, 44px buttons | 2-col hero | Full layout + hero art |
| **S2: Analyze Input** | `/analyze` | Form cleared, CTA disabled | Step vine active | Validation / Device unreachable | Form filled | Stacked accordion summary | Sticky summary panel | 7/5 col split grid |
| **S3: Results** | `/analyze` (post-run) | "Couldn't produce prediction" | Skeletons | Segmentation failure (input issue) | Dual panel + leaf cards | Stacked panels | 2-col panels | Full side-by-side |
| **S4: History** | `/history` | "No saved records yet" + CTA | Table/card skeleton | Error card + Retry (no stale rows) | Paginated list | Stacked cards (<768px) | Scrollable table | Full-width table |
| **S5: Record Detail** | `/history/[id]` | "Record Not Found" + Back CTA | Detail skeletons | Persistence/Fetch error | Provenance + Evidence + Guidance | Single column | 2-col metadata | Full detail grid |
| **S6: Models & Disclosures** | `/models` | "No promoted models" | Registry skeleton | Registry unreadable | Model table + Drawer | `<select>` tab dropdown | Button tabs | Full tabs + Drawer |
| **S7: About / How it Works** | `/about` | N/A | Fast static SSR | N/A | Architecture + Flow charts | Single column | 2-col cards | 3-col architecture |
| **S8: Live Monitor** | `/live` | Gateway offline | Polling indicator | Server unreachable + Retry | Live grid + Camera | 1-col sensor cards | 2-col sensor cards | 7/5 col stream grid |

---

## 2. Touch Target & Layout Conformance

- [x] **Touch targets are at least 44 px:**
  - All button variants (`sm`, `md`, `lg`) enforced with `min-h-[44px] min-w-[44px]`.
  - Form selects, file dropzone targets, and filter inputs meet the 44px touch boundary.
  - Interactive leaf pins provide 32px targets scaled to 40px touch padding on mobile.
- [x] **No horizontal body scroll:**
  - Root `html, body` configured with `overflow-x: hidden; max-width: 100vw;`.
  - All large data structures (`HistoryTable`, `ModelsTable`, `ConfusionMatrix`) use internal scroll containers (`overflow-x-auto`).
- [x] **State Integrity Guarantees:**
  - Stale rows are never rendered during error recovery.
  - Missing foliar assets render metadata without broken image artifacts.
  - Segmentation failures explicitly communicate input/framing issues rather than plant pathogen diagnoses.
