# 💎 Master Frontend Specification & Stitch Prompt
### *For Enterprise Conversational RAG Knowledge Engine*
*Design System: "Kinetic Enterprise" (Attio Ghost UI, Geist Typography & NotebookLM Multi-Source Studio)*

---

## 🎯 Purpose
This document provides the **harmonized, copy-pasteable Master Prompt** for AI UI generators (**Google Stitch, v0.dev, Lovable, Claude 3.7**). It aligns with the generated **Kinetic Enterprise** HTML prototype (using `Geist` font, `Material Symbols Outlined`, 20px dot-matrix atmospheric depth, and the streamlined 3-route navigation: Studio, Documents, Models).

---

# 📋 The Master Stitch Prompt (Copy & Paste Ready)

```markdown
You are an elite Staff Product Designer & Full-Stack Frontend Architect specializing in high-density, industrial-grade B2B SaaS interfaces (like Attio, Linear, Raycast, and NotebookLM).

Design and build the complete, production-grade web application for "KnowledgeEngine" — an Enterprise Conversational Knowledge Engine & Agentic RAG Platform.

> ⚠️ **CRITICAL ARCHITECTURE REQUIREMENT (NO PLACEHOLDERS)**: All 3 core views (Studio, Documents, Models) and the 404 Error Screen must be 100% fully interactive, responsive, and functional with real state management.
> 📱 **Mobile Forward-Compatibility**: Structure all components, TypeScript data interfaces, design tokens, and state stores so they seamlessly translate to an **Expo (SDK 52+) / React Native** mobile application.

---

### 🎨 Design System: "Kinetic Enterprise" (Dual-Theme Specifications):

#### 1. Typography & Hierarchy (Google Fonts: Geist)
- `display-lg`: 72px / 80px (Letter spacing: -0.04em, Weight: 600)
- `headline-xl`: 48px / 56px (Letter spacing: -0.03em, Weight: 600)
- `headline-lg`: 32px / 40px (Letter spacing: -0.02em, Weight: 600)
- `body-lg`: 18px / 28px (Weight: 400)
- `body-md`: 16px / 24px (Weight: 400)
- `label-md`: 14px / 20px (Weight: 500)
- `label-sm`: 12px / 16px (Letter spacing: 0.02em, Weight: 500)
- `font-mono`: JetBrains Mono (for numerical scores, hashes, and telemetry)

#### 2. Atmospheric Background
- **Light Theme**: `#FFFFFF` canvas with `radial-gradient(#e5e7eb 1px, transparent 1px)` (20px 20px grid size)
- **Dark Theme (`class="dark"`)**: `#141313` canvas with `radial-gradient(#36343b 1px, transparent 1px)` (20px 20px grid size)

#### 3. Color Token System
- **Light Mode Tokens**:
  - `surface`: `#FDF8F8` | `surface-container-lowest`: `#FFFFFF` | `surface-container-low`: `#F7F3F2`
  - `primary`: `#000000` | `on-primary`: `#FFFFFF`
  - `secondary`: `#0058BE` (Retrieval / Telemetry Cobalt)
  - `outline`: `#747878` | `outline-variant`: `#C4C7C7`
  - `on-surface`: `#1C1B1B` | `on-surface-variant`: `#444748`
  - `error`: `#BA1A1A` | `error-container`: `#FFDAD6`
- **Dark Mode Tokens (`class="dark"`)**:
  - `surface`: `#141313` | `surface-container-lowest`: `#0F0D13` | `surface-container-low`: `#1D1B20`
  - `primary`: `#D0BCFF` | `on-primary`: `#381E72`
  - `on-surface`: `#E2E2E9` | `on-surface-variant`: `#CAC4D0`
  - `outline`: `#938F99` | `outline-variant`: `#49454F`

#### 4. Iconography
- Use **Material Symbols Outlined** (e.g., `search`, `notifications`, `settings`, `keyboard_command_key`, `description`, `tune`, `check_circle`, `error`, `delete`, `open_in_new`).

---

### 🧭 Global Header (TopNavBar Component):
Sticky header (`w-full h-16 border-b border-outline-variant dark:border-outline bg-surface/90 dark:bg-surface-container-lowest/90 backdrop-blur-md px-margin-desktop z-50`):
- **Brand Title**: `<a href="#">KnowledgeEngine</a>` (`font-headline-lg font-bold text-primary dark:text-on-surface`).
- **Navigation Tabs (Streamlined 3 Core Views)**:
  1. `Studio` (`/`): Active state indicator with bottom border highlight.
  2. `Documents` (`/documents`): Knowledge vault & chunk inventory.
  3. `Models` (`/models`): Inference engine & parameter telemetry.
- **Utility Actions (Right)**:
  - Search input with leading `search` icon and `⌘K` keyboard hint.
  - Notifications button with unread dot.
  - Settings trigger icon.
  - User avatar thumbnail.

---

## 📄 THE 4 COMPLETE SCREEN SPECIFICATIONS

### SCREEN 1: Knowledge Studio (`/`) — 3-Panel Multi-Source Workbench

#### A. Left Panel: Sources Guide & History (280px fixed width):
- **Header**: "Sources (3 selected)" with a global "Select All" toggle checkbox.
- **Source Selection Checklist**: Checkboxes next to each document (`financial_report.pdf`, `system_architecture.md`, `specs_v2.docx`) to dynamically filter grounding scope.
- **Document Card Metadata**: Type icon, file name, chunk badge ("42 chunks"), file size, and deletion action.
- **Upload Zone**: Dotted drag-and-drop box for instant file ingest.
- **Session History List**: Multi-turn chat sessions grouped by date with `/clear` memory reset.

#### B. Center Panel: Conversational Knowledge Stream (Fluid flex-1):
- **Empty State Hero**:
  - Title: "Grounded Knowledge Studio" with subtitle: "Explore your documents with zero hallucinations."
  - 4 Quick Exploration Starter Cards:
    * 📋 "Generate Executive Briefing Doc"
    * 💡 "Extract Key Takeaways & Facts"
    * ❓ "Create FAQ Guide from Documents"
    * 🔍 "Deep Dive: Compare Technical Methodologies"
- **Message Stream (Multi-Turn Chat)**:
  - **User Message**: High-contrast card with timestamp.
  - **Telemetry Micro-Bar**: `[⚡ 182ms | Hybrid RRF: 15 chunks | ONNX Rerank: Top 3 | Confidence: 94.2% | Groq LPU]`
  - **Assistant Message**: Rich Markdown formatting with **interactive inline citation badges** `[1]`, `[2]`.
  - **Inline Citation Badges**: Pill badges (`bg-secondary/10 text-secondary dark:text-primary-fixed border border-secondary/20 hover:bg-secondary/20 cursor-pointer`). Clicking opens the Right Inspector.
  - **Footer Guardrail Status**: "🛡️ 5-Layer Safety Verified • Zero Hallucination Gate Passed".
- **Bottom Prompt Dock**:
  - Floating search dock with auto-resizing textarea and `⌘ + Enter` submit hint.
  - Parameter chips: "Include Full Parents", "Model: gpt-oss-120b", "Confidence: θ > 0.20".

#### C. Right Panel: Studio Notes & Citation Inspector (340px collapsible drawer):
- **Dual-Tab Switcher**:
  - **Tab 1: Citations & Context**: Opens when clicking citation `[1]`. Shows source metadata (`manual.pdf`, Page 4, SHA-256), radial Grounding Confidence Gauge (`94.8%`), exact verbatim quote box, and full 2,000c parent paragraph accordion.
  - **Tab 2: Studio Notes Canvas**: "Pin from Chat" button, rich text note editor, and "Export to Markdown/PDF" buttons.

---

### SCREEN 2: Documents & Knowledge Vault (`/documents`)
- **Header KPI Metric Cards (4 Cards)**:
  1. Total Indexed Documents (`12 Files`)
  2. Total Child Chunks in PostgreSQL (`1,482 Chunks`)
  3. Total Storage Used (`14.2 MB`)
  4. Real-time Ingestion Watcher Status (`Active • <5ms hash-skip`)
- **Document Inventory Table**:
  - Columns: File Name & Type, Date Uploaded, File Size, Total Chunks, SHA-256 Hash status pill (`Verified`), Actions (Inspect Chunks, Re-index, Download, Delete).
- **Chunk Inspector Slide-Over**:
  - Slide-over drawer displaying all child vectors (384d), parent paragraphs, and PostgreSQL vector IDs for the selected file.

---

### SCREEN 3: Inference Engine & Model Configuration (`/models`)
- **Active Model Selector**:
  - Model cards: `Groq openai/gpt-oss-120b` (Default • 800 tok/sec), `Groq llama-3.3-70b-versatile`, `Groq llama-3.1-8b-instant`.
- **Interactive Parameter Sliders**:
  - **Temperature**: Float slider from `0.0` to `1.0`.
  - **Zero-Hallucination Gate ($\theta$)**: Float slider from `0.00` to `1.00` (Default `0.20`).
  - **Retrieval Top-K & Rerank Top-N**: Sliders for $K=15$ and $N=3$.
  - **Hierarchical Parent Context Toggle**: Switch between passing 2,000c full parents vs. 400c child chunks.
- **Safety Toggles**: Prompt Injection Guard, Homoglyph Normalizer, and Secret Redactor.
- **Save Action**: Persists settings with a live toast confirmation.

---

### SCREEN 4: Dedicated 404 Error Page (`/not-found`)
- **Atmospheric Background**: 20px dot-matrix grid pattern (`radial-gradient`).
- **Telemetry Badge**: `STATUS: 404_NOT_FOUND • HASH: 0x00000000` with an animated pulse dot.
- **Headline**: "Resource Not Indexed or Relocated"
- **Description**: "The requested document or knowledge sector could not be found. It may have been archived or is awaiting re-indexing."
- **Action Buttons**:
  1. Primary: `Return to Studio`
  2. Secondary: `Inspect Knowledge Vault`
  3. Tertiary: `[⌘] [K]` command palette launcher.
- **Geometric Decorative Graphic**: Abstract floating minimalist 3D rendering with overlaid "404" typography.

---

### 📦 TypeScript Interfaces & Data Contracts:

```typescript
export interface Telemetry {
  latency_ms: number;
  retrieved_chunks: number;
  reranked_top_n: number;
  top_confidence_score: number;
  llm_backend: string;
  model: string;
}

export interface CitationSource {
  index: number;
  chunk_id: string;
  source: string;
  page?: number;
  confidence_score: number;
  exact_quote: string;
  parent_context: string;
}

export interface StudioNote {
  id: string;
  title: string;
  content: string;
  source_citations?: string[];
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  telemetry?: Telemetry;
  citations?: CitationSource[];
  guardrail_refusal?: boolean;
  timestamp: string;
}

export interface IndexedDocument {
  id: string;
  filename: string;
  extension: 'pdf' | 'docx' | 'pptx' | 'txt' | 'md';
  size_bytes: number;
  total_chunks: number;
  sha256_hash: string;
  uploaded_at: string;
  selected_for_query: boolean;
}

export interface EngineConfig {
  model: string;
  temperature: number;
  confidence_threshold: number;
  top_k: number;
  rerank_top_n: number;
  use_hierarchical_parents: boolean;
  guardrails_enabled: boolean;
  theme: 'light' | 'dark';
}
```
```
