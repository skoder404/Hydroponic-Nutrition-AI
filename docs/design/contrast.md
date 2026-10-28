# WCAG 2.1 AA Colour Contrast Audit (T7-002)

> **Status:** Verified & Compliant · **Target:** WCAG 2.1 Level AA  
> **Standards:** Normal text $\ge 4.5:1$, Large text $\ge 3.0:1$, UI components and focus rings $\ge 3.0:1$.  
> **Formula:** W3C Relative Luminance $L = 0.2126 \cdot R_{lin} + 0.7152 \cdot G_{lin} + 0.0722 \cdot B_{lin}$;  
> Contrast Ratio $= (L_1 + 0.05) / (L_2 + 0.05)$ where $L_1 \ge L_2$.

---

## 1. Executive Summary

Every token pair in the Hydroponic Nutrition AI design system (`docs/DESIGN.md`, `frontend/app/globals.css`) was evaluated against WCAG 2.1 AA specifications.

- **Initial Finding:** `--clay` (`#A4502E`) on `--sand` panels (`#EFE6D2`) achieved $4.49:1$, marginally below the required $4.50:1$ threshold for body text.
- **Remediation:** `--clay` was deepened to `#A24E2D` (a subtle perceptual shift preserving its warm terracotta tone), increasing its contrast to **4.62:1** on `--sand`, **5.31:1** on `--bg`, and **5.73:1** on `--surface` / white text.
- **Final Result:** **100% of text pairs meet or exceed 4.5:1** on all supported backgrounds (`--bg`, `--surface`, `--sand`). All interactive UI components (borders, focus rings) meet or exceed **3.0:1**. Decorative tokens (`--sage`, `--border`) are explicitly designated non-text.

---

## 2. Text on Backgrounds (Target $\ge 4.5:1$)

### 2.1 Text on Cream Page Background (`--bg`: `#FAF6EC`)

| Token | Name / Role | Hex | Ratio | WCAG AA Status |
|---|---|---|---|---|
| `--forest` | Headings, primary text, brand | `#2F4A33` | **9.06:1** | Pass (AAA) |
| `--bark` | Default body text | `#3E3022` | **11.79:1** | Pass (AAA) |
| `--stone` | Secondary / metadata text | `#665C4D` | **6.08:1** | Pass (AA) |
| `--moss` | Links, secondary accents | `#4C6B3C` | **5.60:1** | Pass (AA) |
| `--clay` | Warm accent / CTA text | `#A24E2D` | **5.31:1** | Pass (AA) |
| `--water` | Sensor telemetry accent | `#2F6F7E` | **5.26:1** | Pass (AA) |
| `--success` | Positive status text | `#2E6B3A` | **5.93:1** | Pass (AA) |
| `--warning` | Caution / moderate tier text | `#8A5A00` | **5.49:1** | Pass (AA) |
| `--error` | Alert and error text | `#A32A22` | **6.68:1** | Pass (AA) |
| `--info` | Informational text | `#2A5F8F` | **6.21:1** | Pass (AA) |

---

### 2.2 Text on White Card Surface (`--surface`: `#FFFFFF`)

| Token | Name / Role | Hex | Ratio | WCAG AA Status |
|---|---|---|---|---|
| `--forest` | Headings, primary text | `#2F4A33` | **9.78:1** | Pass (AAA) |
| `--bark` | Body text | `#3E3022` | **12.72:1** | Pass (AAA) |
| `--stone` | Secondary text | `#665C4D` | **6.56:1** | Pass (AA) |
| `--moss` | Interactive links | `#4C6B3C` | **6.04:1** | Pass (AA) |
| `--clay` | Warm text | `#A24E2D` | **5.73:1** | Pass (AA) |
| `--water` | Sensor readings | `#2F6F7E` | **5.68:1** | Pass (AA) |
| `--success` | Success text | `#2E6B3A` | **6.40:1** | Pass (AA) |
| `--warning` | Caution text | `#8A5A00` | **5.93:1** | Pass (AA) |
| `--error` | Error text | `#A32A22` | **7.21:1** | Pass (AAA) |
| `--info` | Info text | `#2A5F8F` | **6.71:1** | Pass (AA) |

---

### 2.3 Text on Sand Panels and Chips (`--sand`: `#EFE6D2`)

| Token | Name / Role | Hex | Ratio | WCAG AA Status |
|---|---|---|---|---|
| `--forest` | Headings and bold chips | `#2F4A33` | **7.88:1** | Pass (AAA) |
| `--bark` | Body text on panels | `#3E3022` | **10.25:1** | Pass (AAA) |
| `--stone` | Secondary text on panels | `#665C4D` | **5.28:1** | Pass (AA) |
| `--moss` | Links / accent chips | `#4C6B3C` | **4.87:1** | Pass (AA) |
| `--clay` | Warm accent (adjusted) | `#A24E2D` | **4.62:1** | Pass (AA) |
| `--water` | Telemetry chip text | `#2F6F7E` | **4.58:1** | Pass (AA) |
| `--success` | Verified chip text | `#2E6B3A` | **5.16:1** | Pass (AA) |
| `--warning` | Moderate tier chip text | `#8A5A00` | **4.78:1** | Pass (AA) |
| `--error` | Alert chip text | `#A32A22` | **5.81:1** | Pass (AA) |
| `--info` | Info chip text | `#2A5F8F` | **5.40:1** | Pass (AA) |

---

## 3. White Text on Solid Buttons and Badges (Target $\ge 4.5:1$)

Text colour `#FFFFFF` on solid background fills:

| Background Token | Fill Hex | White Text Contrast | WCAG AA Status |
|---|---|---|---|
| `--forest` | `#2F4A33` | **9.78:1** | Pass (AAA) |
| `--bark` | `#3E3022` | **12.72:1** | Pass (AAA) |
| `--error` | `#A32A22` | **7.21:1** | Pass (AAA) |
| `--info` | `#2A5F8F` | **6.71:1** | Pass (AA) |
| `--success` | `#2E6B3A` | **6.40:1** | Pass (AA) |
| `--moss` | `#4C6B3C` | **6.04:1** | Pass (AA) |
| `--warning` | `#8A5A00` | **5.93:1** | Pass (AA) |
| `--clay` | `#A24E2D` | **5.73:1** | Pass (AA) |
| `--water` | `#2F6F7E` | **5.68:1** | Pass (AA) |

---

## 4. UI Components, Focus Rings, and Borders (Target $\ge 3.0:1$)

Non-text interactive elements evaluated against adjacent backgrounds:

| Element | Token | Hex | Background | Ratio | WCAG AA Status |
|---|---|---|---|---|---|
| Focus Ring | `--ring-color` | `#4C6B3C` | `--bg` (`#FAF6EC`) | **5.60:1** | Pass ($\ge 3:1$) |
| Focus Ring | `--ring-color` | `#4C6B3C` | `--surface` (`#FFFFFF`) | **6.04:1** | Pass ($\ge 3:1$) |
| Focus Ring | `--ring-color` | `#4C6B3C` | `--sand` (`#EFE6D2`) | **4.87:1** | Pass ($\ge 3:1$) |
| Input Border | `--border-strong` | `#857A67` | `--bg` (`#FAF6EC`) | **3.91:1** | Pass ($\ge 3:1$) |
| Input Border | `--border-strong` | `#857A67` | `--surface` (`#FFFFFF`) | **4.22:1** | Pass ($\ge 3:1$) |
| Input Border | `--border-strong` | `#857A67` | `--sand` (`#EFE6D2`) | **3.40:1** | Pass ($\ge 3:1$) |

---

## 5. Non-Text / Decorative Tokens

Per WCAG 2.1 Guideline 1.4.11, decorative elements that do not convey information or identify component boundaries are exempt from minimum contrast ratios:

- `--border` (`#D9CDB1`): Subtle card dividers ($1.46:1$). Functional inputs rely on `--border-strong` ($3.91:1$) or focus ring outline ($5.60:1$).
- `--sage` (`#8FA77A`): Strictly designated for leaf artwork illustrations and organic fills ($2.44:1$). Never used for text or interactive borders.

---

## 6. Audit Conclusion

All token combinations deployed in the UI satisfy WCAG 2.1 AA requirements. Color is never used as the sole conveyor of information (all statuses are accompanied by descriptive text and iconography per DESIGN.md §Accessibility Requirements).
