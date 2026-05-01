"""Gradio web interface for conversation generation.

Run with:
    python webapp.py

Generates batches of synthetic two-speaker dialogue. Backend is selectable:
'hf' for local transformers, 'openai' for any OpenAI-compatible HTTP server
(LM Studio, Ollama, vLLM, OpenAI itself).

Environment variables for default values:
- CDG_BACKEND      e.g. "openai" or "hf"
- CDG_BASE_URL     e.g. "http://localhost:11434/v1"
- CDG_MODEL_ID     e.g. "llama3.2:1b"
"""

from __future__ import annotations

import os
import tempfile
import time

import gradio as gr

from conversation_dataset_generator.cli import (
    _dedup_check, _load_dedup_model, build_backend, build_role_mapping,
)
from conversation_dataset_generator.generation import (
    generate_args_from_brief_safe, generate_conversation, generate_topic_variation,
)
from conversation_dataset_generator.output import write_jsonl


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #f4efe5;
    --ink: #16140f;
    --ink-soft: #4a4538;
    --rule: #d8cfbb;
    --rule-soft: #e6dfcc;
    --accent: #c2410c;
    --accent-soft: #fde4d3;
    --serif: 'Instrument Serif', 'Iowan Old Style', Georgia, serif;
    --sans: 'IBM Plex Sans', system-ui, sans-serif;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
}

/* override gradio CSS variables that leak through theme */
.gradio-container,
.gradio-container * {
    --color-accent: var(--accent) !important;
    --color-accent-soft: var(--accent-soft) !important;
    --background-fill-primary: var(--paper) !important;
    --background-fill-secondary: var(--paper) !important;
    --body-background-fill: var(--paper) !important;
    --body-text-color: var(--ink) !important;
    --input-background-fill: var(--paper) !important;
    --input-background-fill-focus: var(--paper) !important;
    --block-background-fill: var(--paper) !important;
    --panel-background-fill: var(--paper) !important;
    --section-header-text-color: var(--ink-soft) !important;
}

/* Hide Gradio's default footer (api, settings, built-with) */
footer.svelte-zxu34v,
.gradio-container > footer,
.gradio-container .footer { display: none !important; }
.gradio-container .api-docs-button { display: none !important; }

/* Nuke any Gradio component box backgrounds that aren't paper */
.gradio-container .gr-group,
.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .gr-block,
.gradio-container .gr-panel,
.gradio-container fieldset,
.gradio-container .form,
.gradio-container .block,
.gradio-container [class*="svelte-"][class*="block"],
.gradio-container [class*="svelte-"][class*="form"] {
    background: transparent !important;
    background-color: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Constrain Gradio's main wrapper so the dashboard fits the viewport */
.gradio-container > main,
.gradio-container > .main,
main.contain,
.gradio-container > div > main {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    max-height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
main.contain > div,
main.contain > .column {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}
/* Only the OUTER grid row should stretch — not every form row */
main.contain > div > .row,
.gradio-container #cdg-grid {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
}

/* Override Gradio prose colors that color loading text indigo */
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .md,
.gradio-container .md * {
    color: var(--ink) !important;
}
.gradio-container .cdg-loading,
.gradio-container .cdg-loading * { color: var(--ink-soft) !important; }
.gradio-container .cdg-error,
.gradio-container .cdg-error * { color: var(--accent) !important; }
.gradio-container .cdg-success,
.gradio-container .cdg-success strong { color: var(--ink) !important; }
.gradio-container .cdg-success strong { color: var(--accent) !important; }

/* ---------------- global reset of Gradio chrome ---------------- */
html, body, .gradio-container, .gradio-container * { box-sizing: border-box; }
html, body { margin: 0 !important; padding: 0 !important; height: 100% !important; }
body { background: var(--paper) !important; color: var(--ink) !important; font-family: var(--sans) !important; }

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    display: flex !important;
    flex-direction: column !important;
}
.gradio-container > .main { flex: 1; min-height: 0; display: flex; flex-direction: column; }

/* Strip every default border, background, and shadow off Gradio blocks */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-block,
.gradio-container .gr-panel {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

/* Inputs: underline only, no box */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container input[type="number"],
.gradio-container textarea {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 6px 0 !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    color: var(--ink) !important;
    transition: border-color 160ms ease;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    outline: none !important;
    border-bottom-color: var(--accent) !important;
}

/* Labels: small caps, sans, muted */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .gr-input-label {
    font-family: var(--sans) !important;
    font-size: 10.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 500 !important;
    color: var(--ink-soft) !important;
    margin-bottom: 6px !important;
}

/* Info text */
.gradio-container .info {
    font-family: var(--sans) !important;
    font-size: 11px !important;
    font-style: italic !important;
    color: var(--ink-soft) !important;
    opacity: 0.75;
    margin-top: 4px !important;
}

/* Radio: pill-less, just clickable text with accent underline when selected */
.gradio-container .wrap.svelte-1ipelgc,
.gradio-container fieldset {
    border: 0 !important;
    background: transparent !important;
    padding: 0 !important;
}
.gradio-container input[type="radio"] {
    accent-color: var(--accent) !important;
}

/* Sliders: thin track, accent thumb */
.gradio-container input[type="range"] {
    accent-color: var(--accent) !important;
}

/* Checkboxes */
.gradio-container input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* Accordion: strip box, just left rule */
.gradio-container .accordion,
.gradio-container details {
    background: transparent !important;
    border: 0 !important;
    border-left: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    padding: 0 0 0 14px !important;
    margin: 12px 0 !important;
}
.gradio-container summary {
    font-family: var(--sans) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--ink-soft) !important;
    cursor: pointer;
    padding: 4px 0 !important;
}

/* Generate button: solid black slab, mono caps */
#cdg-generate-btn,
#cdg-generate-btn button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: 0 !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    padding: 14px 18px !important;
    margin-top: 16px !important;
    cursor: pointer;
    transition: background 160ms ease, transform 160ms ease;
}
#cdg-generate-btn button:hover { background: var(--accent) !important; }
#cdg-generate-btn button:active { transform: translateY(1px); }

/* Brainstorm button: smaller, ink outline */
#cdg-brief-row { margin: 8px 0 4px 0; align-items: end; gap: 8px; }
#cdg-brainstorm-btn,
#cdg-brainstorm-btn button {
    background: transparent !important;
    color: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 8px 12px !important;
    cursor: pointer;
    transition: background 160ms ease, color 160ms ease;
    align-self: end !important;
}
#cdg-brainstorm-btn button:hover {
    background: var(--ink) !important;
    color: var(--paper) !important;
}

/* Auto-fix button: vermillion outline, secondary tone */
#cdg-fix-row { margin: 8px 0 4px 0; align-items: center; }
#cdg-fix-btn,
#cdg-fix-btn button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 8px 14px !important;
    cursor: pointer;
    width: auto !important;
    transition: background 160ms ease, color 160ms ease;
}
#cdg-fix-btn button:hover {
    background: var(--accent) !important;
    color: var(--paper) !important;
}
#cdg-fix-status {
    font-family: var(--sans);
    font-size: 12px;
    color: var(--ink-soft);
    padding: 0 12px !important;
}

/* File component: minimal */
.gradio-container .file-preview,
.gradio-container [data-testid="file"] {
    background: transparent !important;
    border: 1px dashed var(--rule) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ---------------- dashboard layout ---------------- */
#cdg-mast {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 28px;
    border-bottom: 1px solid var(--ink);
    flex-shrink: 0;
    height: 56px;
}
#cdg-mast .title {
    font-family: var(--serif);
    font-size: 26px;
    font-weight: 400;
    letter-spacing: -0.01em;
    line-height: 1;
    color: var(--ink);
}
#cdg-mast .title em {
    font-style: italic;
    color: var(--accent);
}
#cdg-mast .meta {
    font-family: var(--mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink-soft);
}

/* The grid: form on left, dashboard on right */
#cdg-grid {
    display: grid !important;
    grid-template-columns: minmax(360px, 400px) minmax(0, 1fr);
    flex: 1;
    min-height: 0;
    height: 100%;
}
#cdg-grid > * { min-height: 0; height: 100%; }

.cdg-pane {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 18px 24px 24px;
    height: 100% !important;
    flex-wrap: nowrap !important;
    flex-direction: column !important;
    display: flex !important;
}
.cdg-pane > * { flex-shrink: 0 !important; flex-grow: 0 !important; }
.cdg-pane.left { border-right: 1px solid var(--ink); }

/* Right pane: scrolls internally, no card chrome */
#cdg-right.cdg-pane { overflow-y: auto !important; padding: 18px 24px 24px; height: 100% !important; }

/* Section heading: sectional rule + serif label */
.cdg-section {
    margin: 0 0 16px 0;
}
.cdg-section:first-child { margin-top: 0; }
.cdg-section .heading {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin: 0 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--rule-soft);
}
.cdg-section .heading .num {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--accent);
    letter-spacing: 0.18em;
}
.cdg-section .heading .name {
    font-family: var(--serif);
    font-size: 16px;
    font-style: italic;
    color: var(--ink);
}

/* Status / metrics / preview wrappers — no card chrome, just typography */
#cdg-status, #cdg-metrics, #cdg-preview {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    font-family: var(--sans) !important;
    color: var(--ink) !important;
}

/* Metrics: headline + stat block — each stat shows TARGET vs SEEING */
#cdg-metrics .headline {
    font-family: var(--serif);
    font-size: 16px;
    font-style: italic;
    line-height: 1.4;
    padding: 8px 12px;
    margin: 2px 0 12px 0;
    border-left: 3px solid var(--rule);
}
#cdg-metrics .headline strong {
    font-style: normal;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-right: 6px;
}
#cdg-metrics .headline.status-ok    { border-left-color: #2d6a4f; color: var(--ink); }
#cdg-metrics .headline.status-ok strong    { color: #2d6a4f; }
#cdg-metrics .headline.status-warn  { border-left-color: #b45309; color: var(--ink); }
#cdg-metrics .headline.status-warn strong  { color: #b45309; }
#cdg-metrics .headline.status-alert { border-left-color: var(--accent); color: var(--ink); }
#cdg-metrics .headline.status-alert strong { color: var(--accent); }

#cdg-metrics .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px 22px;
    margin: 6px 0 10px 0;
}
#cdg-metrics .stat {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding-left: 10px;
    border-left: 2px solid var(--rule-soft);
}
#cdg-metrics .stat.status-ok    { border-left-color: #2d6a4f; }
#cdg-metrics .stat.status-warn  { border-left-color: #b45309; }
#cdg-metrics .stat.status-alert { border-left-color: var(--accent); }
#cdg-metrics .stat .v {
    font-family: var(--serif);
    font-size: 28px;
    line-height: 1;
    color: var(--ink);
    font-feature-settings: "tnum" 1;
}
#cdg-metrics .stat.status-alert .v { color: var(--accent); }
#cdg-metrics .stat.status-warn  .v { color: #b45309; }
#cdg-metrics .stat .k {
    font-family: var(--mono);
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-soft);
    margin-top: 4px;
}
#cdg-metrics .stat .target {
    font-family: var(--mono);
    font-size: 9.5px;
    color: var(--ink-soft);
    opacity: 0.75;
    margin-top: 1px;
}
#cdg-metrics .stat.status-ok .target    { color: #2d6a4f; opacity: 0.9; }
#cdg-metrics .stat.status-warn .target  { color: #b45309; opacity: 0.9; }
#cdg-metrics .stat.status-alert .target { color: var(--accent); opacity: 0.9; }
#cdg-metrics .summary-line {
    font-family: var(--mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-soft);
    border-top: 1px solid var(--rule-soft);
    padding-top: 8px;
    margin-top: 6px;
}
#cdg-metrics .recs {
    margin-top: 14px;
    padding: 12px 14px;
    border-left: 3px solid var(--accent);
    background: var(--accent-soft);
}
#cdg-metrics .recs-label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--accent);
    margin-bottom: 8px;
}
#cdg-metrics .recs ul {
    margin: 0;
    padding-left: 18px;
    font-family: var(--sans);
    font-size: 13px;
    color: var(--ink);
}
#cdg-metrics .recs li {
    margin: 4px 0;
    line-height: 1.5;
}

/* Preview turns: editorial dialogue */
#cdg-preview { font-size: 14px; line-height: 1.7; }
#cdg-preview h3 {
    font-family: var(--serif);
    font-size: 20px;
    font-weight: 400;
    margin: 22px 0 10px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--rule-soft);
    color: var(--ink);
}
#cdg-preview h3:first-child { margin-top: 0; }
#cdg-preview p { margin: 6px 0; }
#cdg-preview strong {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
    margin-right: 8px;
}
#cdg-preview em {
    font-family: var(--serif);
    font-style: italic;
    color: var(--ink-soft);
    font-size: 14px;
}

/* Status flavors */
.cdg-error, .cdg-success, .cdg-loading {
    font-family: var(--mono);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 12px 0;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
}
.cdg-error { color: var(--accent); border-color: var(--accent); }
.cdg-error strong { color: var(--accent); margin-right: 10px; }
.cdg-success { color: var(--ink); }
.cdg-success strong { color: var(--accent); margin-right: 10px; }
.cdg-loading {
    color: var(--ink-soft);
    font-style: italic;
    text-transform: none;
    letter-spacing: 0;
    font-family: var(--serif);
    font-size: 15px;
    padding: 4px 0 !important;
    background: transparent !important;
    border: 0 !important;
}

/* Override container heading element when used as section divider in the panes */
.gr-prose, .gr-prose h1, .gr-prose h2, .gr-prose h3 { margin: 0 !important; }

.cdg-error {
    padding: 16px 20px;
    border-radius: var(--cdg-radius);
    background: rgba(220, 38, 38, 0.08);
    border: 1px solid rgba(220, 38, 38, 0.35);
    color: #b91c1c;
    font-size: 14px;
    line-height: 1.55;
}
.cdg-error strong {
    color: #991b1b;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 12px;
}
.cdg-loading {
    padding: 18px 22px;
    border-radius: var(--cdg-radius);
    background: rgba(99, 102, 241, 0.08);
    border: 1px dashed rgba(99, 102, 241, 0.45);
    color: #4f46e5;
    font-style: italic;
    font-size: 14px;
}
.cdg-success {
    padding: 16px 20px;
    border-radius: var(--cdg-radius);
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #047857;
    font-size: 14px;
    line-height: 1.55;
}
.cdg-success strong {
    color: #065f46;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 12px;
}
"""


# ----------------------------- formatting helpers -----------------------------


def _error_block(message: str) -> str:
    return (
        '<div class="cdg-error">'
        '<strong>Error</strong><br>'
        f'{message}'
        '</div>'
    )


def _success_block(headline: str, body: str = "") -> str:
    extra = f'<br>{body}' if body else ""
    return (
        '<div class="cdg-success">'
        f'<strong>Done</strong><br>{headline}{extra}'
        '</div>'
    )


def _format_preview(conversations: list, limit: int = 3) -> str:
    """Render up to `limit` conversations as markdown."""
    if not conversations:
        return ""
    parts = []
    for idx, conv in enumerate(conversations[:limit], start=1):
        title = conv.get("topic") or f"Conversation {idx}"
        parts.append(f"### {idx}. {title}")
        for turn in conv.get("turns", []):
            speaker = turn.get("speaker_name") or "?"
            parts.append(f"**{speaker}:** {turn.get('value', '')}")
        parts.append("")
    if len(conversations) > limit:
        parts.append(
            f"*…and {len(conversations) - limit} more in the JSONL download.*"
        )
    return "\n\n".join(parts)


_FIX_SYSTEM_PROMPT = (
    "You are a dialogue coach. Your job is to rewrite persona descriptions so "
    "their VOICES will sound orthogonal in dialogue — measurably different on "
    "concrete axes. Pick a different axis for each persona from this list:\n"
    "  - terse, clipped sentences vs. ornate, subordinate clauses\n"
    "  - clinical / technical vocabulary vs. emotional / metaphorical\n"
    "  - decisive declarations vs. hedged questions\n"
    "  - formal register vs. slangy / informal\n"
    "  - skeptical / contrarian vs. enthusiastic / agreeable\n"
    "Each persona MUST land on a DIFFERENT axis from the others. "
    "Keep the character's name and core identity. Avoid flowery prose like "
    "'eyes that never leave the truth' — be concrete about HOW they speak.\n\n"
    "Return ONLY rewritten descriptions, one per line, in EXACTLY this format "
    "with no commentary, headers, or markdown:\n"
    "Name => description"
)


def _fix_personas_messages(personas: list) -> list:
    """Build chat messages for a persona-distinctness rewrite."""
    persona_block = "\n".join(f"- {n}: {d}" for n, d in personas)
    n = len(personas)
    return [
        {"role": "system", "content": _FIX_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"These {n} personas currently sound too similar in dialogue:\n\n"
            f"{persona_block}\n\n"
            f"Rewrite each so they sit on DIFFERENT speech axes. For example, if "
            f"one is terse and clinical, another might be verbose and emotional, "
            f"another slangy and skeptical. Be specific about sentence length, "
            f"vocabulary register, and rhetorical habits. Keep names unchanged.\n\n"
            f"Format: Name => description (one per line, no other text)"
        )},
    ]


def _parse_rewritten_personas(text: str, original_names: list) -> dict:
    """Parse 'Name => description' lines, return dict of original-name → new desc."""
    out = {}
    if not text:
        return out
    name_set = {n.lower(): n for n in original_names}
    for line in text.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if "=>" not in line:
            continue
        name, _, desc = line.partition("=>")
        name = name.strip().rstrip(":").strip()
        desc = desc.strip()
        if name and desc:
            orig = name_set.get(name.lower())
            if orig is not None:
                out[orig] = desc
    return out


def _fix_topic_messages(topic: str) -> list:
    return [
        {"role": "system", "content": (
            "You rewrite conversation topics to be broader and richer, suitable for "
            "generating a diverse batch of dialogues. Keep the spirit of the original. "
            "Return ONE rewritten topic line, no commentary."
        )},
        {"role": "user", "content": (
            f"Current topic: {topic}\n\n"
            "Rewrite it as a broader theme that supports varied conversations. "
            "Return only the new topic, no quotes, no labels."
        )},
    ]


def brainstorm_handler(
    backend_kind: str,
    model_id: str,
    api_base_url: str,
    api_key: str,
    load_in_4bit: bool,
    brief: str,
):
    """Take a one-line creative brief, ask the LLM to invent personas + scene.
    Returns (persona1, persona1_desc, persona2, persona2_desc, topic, scenario,
    style, status_html)."""
    if not brief or not brief.strip():
        return ("", "", "", "", "", "", "",
                _error_block("Brief is empty — paste an idea like 'Sherlock and Watson debate AI'."))

    try:
        backend = build_backend(
            kind=backend_kind, model_id=model_id,
            load_in_4bit=load_in_4bit,
            api_base_url=api_base_url, api_key=api_key or None,
        )
    except Exception as exc:
        return ("", "", "", "", "", "", "",
                _error_block(f"Failed to build backend: {exc}"))

    args = generate_args_from_brief_safe(brief, backend=backend)
    if args is None:
        return ("", "", "", "", "", "", "",
                _error_block("LLM couldn't produce parseable personas. Try rephrasing the brief."))

    return (
        args.get("persona1", ""),
        args.get("persona1_desc", ""),
        args.get("persona2", ""),
        args.get("persona2_desc", ""),
        args.get("topic", ""),
        args.get("scenario", ""),
        args.get("style", ""),
        _success_block(f"Brainstormed personas + scene from your brief. Edit if needed, then click Generate."),
    )


def auto_fix_handler(
    backend_kind: str,
    model_id: str,
    api_base_url: str,
    api_key: str,
    load_in_4bit: bool,
    persona1: str,
    persona1_desc: str,
    persona2: str,
    persona2_desc: str,
    extra_personas: str,
    topic: str,
    scenario: str,
    style: str,
    max_new_tokens: int,
    enable_variation: bool,
    dedup_threshold: float,
    metrics_state,
):
    """Apply every applicable fix based on the latest metrics. Returns:
    (new_p1_desc, new_p2_desc, new_extra_personas, new_topic, new_scenario,
     new_style, new_max_new_tokens, new_enable_variation, status_html)."""
    metrics = metrics_state if isinstance(metrics_state, dict) else {}
    if not metrics or metrics.get("num_conversations", 0) < 2:
        return (
            persona1_desc, persona2_desc, extra_personas, topic, scenario, style,
            max_new_tokens, enable_variation,
            _error_block("No recent metrics — generate a batch first, then click Auto-fix."),
        )

    new_p1_desc = persona1_desc
    new_p2_desc = persona2_desc
    new_extras = extra_personas
    new_topic = topic
    new_scenario = scenario
    new_style = style
    new_max = max_new_tokens
    new_variation = enable_variation
    fixes_applied = []

    # --- non-LLM fixes (always cheap, do first) ---

    if metrics.get("distinct_2", 1.0) < 0.70 and not enable_variation:
        new_variation = True
        fixes_applied.append("enabled topic variation")

    if metrics.get("self_repetition_rate", 0) > 0.10:
        new_max = max(384, int(max_new_tokens * 0.75))
        fixes_applied.append(f"lowered max-new-tokens to {new_max}")

    # --- LLM-driven fixes ---

    needs_persona_rewrite = metrics.get("speaker_distinctiveness", 1.0) < 0.30
    needs_topic_rewrite = metrics.get("topic_diversity", 1.0) < 0.50
    coherence = metrics.get("turn_coherence")
    needs_scene_rewrite = coherence is not None and coherence < 0.30

    if needs_persona_rewrite or needs_topic_rewrite or needs_scene_rewrite:
        try:
            backend = build_backend(
                kind=backend_kind, model_id=model_id,
                load_in_4bit=load_in_4bit,
                api_base_url=api_base_url, api_key=api_key or None,
            )
        except Exception as exc:
            status_text = (
                f"Applied {len(fixes_applied)} non-LLM fix(es). "
                f"Couldn't build backend for the rest: {exc}"
            ) if fixes_applied else f"Failed to build backend: {exc}"
            return (
                new_p1_desc, new_p2_desc, new_extras, new_topic, new_scenario,
                new_style, new_max, new_variation,
                _error_block(status_text),
            )

        if needs_persona_rewrite:
            extras_list = _parse_extra_personas(extra_personas)
            full = [(persona1, persona1_desc or ""), (persona2, persona2_desc or "")] + extras_list
            full = [(n, d) for n, d in full if n.strip()]
            text = backend.complete(_fix_personas_messages(full), max_new_tokens=600, temperature=0.9)
            rewrites = _parse_rewritten_personas(text or "", [n for n, _ in full])
            if rewrites:
                new_p1_desc = rewrites.get(persona1, new_p1_desc)
                new_p2_desc = rewrites.get(persona2, new_p2_desc)
                new_extras = "\n".join(f"{n} | {rewrites.get(n, d)}" for n, d in extras_list)
                fixes_applied.append(f"rewrote {len(rewrites)} persona description(s)")

        if needs_topic_rewrite:
            text = backend.complete(_fix_topic_messages(topic), max_new_tokens=120, temperature=0.9)
            if text:
                cleaned = text.strip().splitlines()[0].strip().strip('"').strip("'")
                if cleaned and cleaned.lower() != topic.lower():
                    new_topic = cleaned
                    fixes_applied.append("broadened topic")

        if needs_scene_rewrite:
            scene_msgs = [
                {"role": "system", "content": (
                    "You sharpen scene descriptions so dialogue feels grounded and "
                    "characters react to the same situation. Return TWO lines: "
                    "Scenario: <new>\nStyle: <new>"
                )},
                {"role": "user", "content": (
                    f"Topic: {topic}\nScenario: {scenario}\nStyle: {style}\n\n"
                    "Rewrite scenario with concrete sensory anchors and constraints, "
                    "and tighten the style guidance."
                )},
            ]
            text = backend.complete(scene_msgs, max_new_tokens=200, temperature=0.8)
            if text:
                lines = [l.strip() for l in text.splitlines() if ":" in l]
                for line in lines:
                    label, _, val = line.partition(":")
                    val = val.strip().strip('"').strip("'")
                    if label.strip().lower().startswith("scenario") and val:
                        new_scenario = val
                    elif label.strip().lower().startswith("style") and val:
                        new_style = val
                if new_scenario != scenario or new_style != style:
                    fixes_applied.append("sharpened scene description")

    if not fixes_applied:
        return (
            new_p1_desc, new_p2_desc, new_extras, new_topic, new_scenario,
            new_style, new_max, new_variation,
            _success_block(
                "Targets already met (or remaining issues are server-side, "
                "like temperature). Nothing to apply."
            ),
        )

    summary = "; ".join(fixes_applied) + ". Re-generate to see the impact."
    return (
        new_p1_desc, new_p2_desc, new_extras, new_topic, new_scenario,
        new_style, new_max, new_variation,
        _success_block(summary),
    )


def fix_personas_handler(
    backend_kind: str,
    model_id: str,
    api_base_url: str,
    api_key: str,
    load_in_4bit: bool,
    persona1: str,
    persona1_desc: str,
    persona2: str,
    persona2_desc: str,
    extra_personas: str,
):
    """Rewrite persona descriptions via the configured LLM. Returns
    (new_persona1_desc, new_persona2_desc, new_extra_personas, status_html)."""
    try:
        backend = build_backend(
            kind=backend_kind, model_id=model_id,
            load_in_4bit=load_in_4bit,
            api_base_url=api_base_url, api_key=api_key or None,
        )
    except Exception as exc:
        return (
            persona1_desc, persona2_desc, extra_personas,
            _error_block(f"Failed to build backend: {exc}"),
        )

    extras = _parse_extra_personas(extra_personas)
    full = [(persona1, persona1_desc or ""), (persona2, persona2_desc or "")]
    full = full + extras
    full = [(n, d) for n, d in full if n.strip()]
    if not full:
        return (
            persona1_desc, persona2_desc, extra_personas,
            _error_block("No personas to rewrite."),
        )

    text = backend.complete(
        _fix_personas_messages(full),
        max_new_tokens=600, temperature=0.9,
    )
    if not text:
        return (
            persona1_desc, persona2_desc, extra_personas,
            _error_block("LLM returned no text — check the server."),
        )

    rewrites = _parse_rewritten_personas(text, [n for n, _ in full])
    if not rewrites:
        return (
            persona1_desc, persona2_desc, extra_personas,
            _error_block("Couldn't parse the rewrite. Try again."),
        )

    new_p1 = rewrites.get(persona1, persona1_desc)
    new_p2 = rewrites.get(persona2, persona2_desc)
    new_extras = "\n".join(
        f"{n} | {rewrites.get(n, d)}" for n, d in extras
    )
    return (
        new_p1, new_p2, new_extras,
        _success_block(f"Rewrote {len(rewrites)} persona description(s). Re-generate to see the impact."),
    )


PRESETS = {
    "— Custom —": None,
    "Sherlock & Watson · AI surveillance": {
        "persona1": "Sherlock Holmes",
        "persona1_desc": "A coldly logical Victorian detective with cutting wit. Speaks in clipped, deductive observations.",
        "persona2": "Dr. Watson",
        "persona2_desc": "A loyal, slightly bewildered army doctor and trusted companion. Asks the questions readers would ask.",
        "extra_personas": "",
        "topic": "the ethics of AI surveillance",
        "scenario": "221B Baker Street, late evening",
        "style": "Tense and dramatic with period-appropriate diction",
    },
    "Avengers · planetary threat": {
        "persona1": "Iron Man",
        "persona1_desc": "Genius billionaire, rapid-fire wit, deflects emotion with humor. Heavy on technical jargon and pop-culture references.",
        "persona2": "Captain America",
        "persona2_desc": "Principled, earnest, old-fashioned politeness. Natural leader who inspires with clear, direct speech.",
        "extra_personas": "Thor | Boisterous Shakespearean formality. Uses 'verily' and refers to humans as 'mortals'.",
        "topic": "how to confront a planetary threat without destroying the city",
        "scenario": "the Avengers Tower conference room",
        "style": "Tense and dramatic, with light banter",
    },
    "Sci-fi crew · anomaly on the bridge": {
        "persona1": "Captain Eva Rostova",
        "persona1_desc": "Experienced, cautious starship captain. Focused on procedure and crew safety. Speaks formally with measured authority.",
        "persona2": "Dr. Aris Thorne",
        "persona2_desc": "Brilliant but impulsive xeno-archaeologist. Eager for discovery, sometimes disregards protocol. Speaks excitedly with technical jargon.",
        "extra_personas": (
            "ARIA | The ship's AI. Precise, literal-minded, occasionally dry. Struggles with human idioms.\n"
            "Chief Engineer Malik | Pragmatic, no-nonsense engineer. Short sentences. Worried about power consumption."
        ),
        "topic": "an unidentified anomaly approaching the ship",
        "scenario": "the bridge of the starship Pioneer",
        "style": "Procedural and tense, with technical jargon",
    },
    "Code review · senior & intern": {
        "persona1": "Ada",
        "persona1_desc": "A pragmatic senior engineer. Explains tradeoffs clearly without condescension. Uses concrete examples.",
        "persona2": "Lin",
        "persona2_desc": "An eager intern asking great clarifying questions. Comfortable saying 'I don't know'.",
        "extra_personas": "",
        "topic": "why monoliths beat microservices for small teams",
        "scenario": "a code review session over a shared screen",
        "style": "Direct and educational",
    },
    "Chefs · is umami overrated": {
        "persona1": "Maya",
        "persona1_desc": "An enthusiastic chef obsessed with umami. Uses food metaphors constantly.",
        "persona2": "Diego",
        "persona2_desc": "A skeptical food scientist. Demands evidence. Slightly contrarian.",
        "extra_personas": "",
        "topic": "whether umami is overrated",
        "scenario": "a steamy professional kitchen during prep",
        "style": "Spirited and friendly with sharp disagreement",
    },
}


def _apply_preset(preset_name: str):
    """Return a tuple of values for (persona1, persona1_desc, persona2, persona2_desc,
    extra_personas, topic, scenario, style). Returns 8 gr.update no-ops when Custom."""
    p = PRESETS.get(preset_name)
    if not p:
        return tuple(gr.update() for _ in range(8))
    return (
        p.get("persona1", ""),
        p.get("persona1_desc", ""),
        p.get("persona2", ""),
        p.get("persona2_desc", ""),
        p.get("extra_personas", ""),
        p.get("topic", ""),
        p.get("scenario", ""),
        p.get("style", ""),
    )


def _parse_extra_personas(text: str) -> list:
    """Parse 'Name | Description' lines into [(name, desc), ...]. Skips blanks/malformed."""
    if not text or not text.strip():
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, _, desc = line.partition("|")
        name = name.strip()
        desc = desc.strip()
        if name:
            out.append((name, desc))
    return out


def _recommendations(metrics: dict) -> list:
    """Plain-English fixes for any metrics that missed their targets."""
    recs = []
    if metrics.get("num_conversations", 0) < 2:
        return recs

    sd = metrics.get("speaker_distinctiveness")
    if sd is not None and sd < 0.30:
        recs.append(
            "Make persona descriptions more contrastive — different age, vocabulary, "
            "or stance (skeptical vs enthusiastic, terse vs verbose)."
        )

    d2 = metrics.get("distinct_2", 1.0)
    if d2 < 0.70:
        recs.append(
            "Enable variation per example, or batch with a wider topic. "
            "Lexical diversity is below target."
        )

    td = metrics.get("topic_diversity")
    if td is not None and td < 0.50:
        recs.append(
            "Conversations are clustered on the same theme. Use creative-brief mode or "
            "broaden the topic each run."
        )

    sr = metrics.get("self_repetition_rate", 0)
    if sr > 0.10:
        recs.append(
            "Model is repeating itself within conversations — lower max-new-tokens, "
            "or shorten the requested style."
        )

    coherence = metrics.get("turn_coherence")
    if coherence is not None:
        if coherence < 0.30:
            recs.append(
                "Coherence low — turns feel unrelated. Try a stronger scene description, "
                "or a smaller temperature on the model server."
            )
        elif coherence > 0.60:
            recs.append(
                "Coherence very high — turns may sound robotic / repetitive. "
                "Try a higher temperature on the model server."
            )

    return recs


def _grade(value: float, low: float, high: float | None = None) -> str:
    """Classify a metric value: 'ok' / 'warn' / 'alert' against a target.

    If high is None: higher is better, target is `value >= low`.
    If high is set: target is `low <= value <= high` (range metric like coherence).
    """
    if high is not None:
        if low <= value <= high:
            return "ok"
        miss = min(abs(value - low), abs(value - high))
        return "warn" if miss < 0.1 else "alert"
    if value >= low:
        return "ok"
    if value >= low * 0.85:
        return "warn"
    return "alert"


def _format_metrics_card(metrics: dict, dedup_drops: int = 0) -> str:
    """Render evaluation metrics as a stat-block with goals + plain-English headline."""
    if not metrics:
        return ""

    n = metrics.get("num_conversations", 0)

    def stat(value: str, label: str, target: str = "", status: str = "neutral") -> str:
        target_html = f'<div class="target">{target}</div>' if target else ""
        return (
            f'<div class="stat status-{status}">'
            f'<div class="v">{value}</div>'
            f'<div class="k">{label}</div>'
            f'{target_html}'
            f'</div>'
        )

    # Compute graded metrics
    d1 = metrics.get("distinct_1", 0.0)
    d2 = metrics.get("distinct_2", 0.0)
    d3 = metrics.get("distinct_3", 0.0)
    ttr = metrics.get("vocabulary_richness", 0.0)
    vendi = metrics.get("vendi_score")
    vendi_ratio = (vendi / n) if (vendi is not None and n) else None
    topic_div = metrics.get("topic_diversity")
    speaker_dist = metrics.get("speaker_distinctiveness")
    coherence = metrics.get("turn_coherence")
    self_rep = metrics.get("self_repetition_rate")

    stats = []
    stats.append(stat(str(n), "conversations"))
    stats.append(stat(str(metrics.get("total_turns", 0)), "total turns"))

    if vendi_ratio is not None:
        stats.append(stat(
            f"{vendi_ratio:.0%}",
            "effective uniqueness",
            "target ≥ 70%",
            _grade(vendi_ratio, 0.70),
        ))
    stats.append(stat(
        f"{d2:.2f}", "distinct-2 (bigrams)",
        "target ≥ 0.70", _grade(d2, 0.70),
    ))
    if topic_div is not None:
        stats.append(stat(
            f"{topic_div:.2f}", "topic diversity",
            "target ≥ 0.50", _grade(topic_div, 0.50),
        ))
    if speaker_dist is not None:
        stats.append(stat(
            f"{speaker_dist:.2f}", "speaker distinctness",
            "target ≥ 0.30", _grade(speaker_dist, 0.30),
        ))
    if coherence is not None:
        stats.append(stat(
            f"{coherence:.2f}", "turn coherence",
            "ideal 0.30–0.60", _grade(coherence, 0.30, 0.60),
        ))
    if self_rep is not None:
        stats.append(stat(
            f"{self_rep:.1%}", "self-repetition",
            "target ≤ 5%", "ok" if self_rep <= 0.05 else "warn" if self_rep <= 0.10 else "alert",
        ))

    grid = '<div class="stat-grid">' + "".join(stats) + "</div>"

    # Plain-English headline
    failures = []
    wins = []
    def check(name, value, target_low, target_high=None):
        g = _grade(value, target_low, target_high)
        if g == "ok": wins.append(name)
        elif g == "alert": failures.append(name)
    if vendi_ratio is not None: check("uniqueness", vendi_ratio, 0.70)
    check("lexical variety", d2, 0.70)
    if topic_div is not None: check("topic spread", topic_div, 0.50)
    if speaker_dist is not None: check("distinct voices", speaker_dist, 0.30)
    if coherence is not None: check("coherence", coherence, 0.30, 0.60)
    if self_rep is not None and self_rep > 0.10:
        failures.append("repetition")

    if not failures and len(wins) >= 3:
        headline_cls = "ok"
        headline_text = f"<strong>Healthy.</strong> Diverse and coherent across {len(wins)} metrics."
    elif failures:
        headline_cls = "alert"
        headline_text = f"<strong>Needs attention:</strong> {', '.join(failures)}."
    else:
        headline_cls = "warn"
        headline_text = "<strong>Mixed.</strong> Some metrics below target — see flags below."

    headline = f'<div class="headline status-{headline_cls}">{headline_text}</div>'

    # Footer line for less-headline metrics
    footer_bits = [
        f"distinct-1 {d1:.2f}",
        f"distinct-3 {d3:.2f}",
        f"ttr {ttr:.2f}",
    ]
    if vendi is not None:
        footer_bits.append(f"vendi {vendi:.2f} / {n}")
    if dedup_drops:
        footer_bits.append(f"dedup drops {dedup_drops}")
    footer = '<div class="summary-line">' + "  ·  ".join(footer_bits) + "</div>"

    recs = _recommendations(metrics)
    if recs:
        items = "".join(f"<li>{r}</li>" for r in recs)
        rec_block = (
            '<div class="recs"><div class="recs-label">How to improve</div>'
            f'<ul>{items}</ul></div>'
        )
    else:
        rec_block = ""

    return headline + grid + footer + rec_block


# ----------------------------- evaluation hook --------------------------------


def _compute_metrics(jsonl_path: str) -> dict:
    """Run evaluation against a JSONL path. Returns empty dict on failure."""
    try:
        from conversation_dataset_generator.evaluation import run_evaluation
        return run_evaluation(jsonl_path)
    except Exception:
        return {}


# ----------------------------- main handler -----------------------------------


def generate_handler(
    backend_kind: str,
    model_id: str,
    api_base_url: str,
    api_key: str,
    load_in_4bit: bool,
    persona1: str,
    persona1_desc: str,
    persona2: str,
    persona2_desc: str,
    topic: str,
    scenario: str,
    style: str,
    max_new_tokens: int,
    num_examples: int,
    enable_variation: bool,
    dedup_threshold: float,
    extra_personas: str = "",
    train_speaker: str = "auto",
    include_points: str = "",
):
    """Generate N conversations and return (status, metrics, preview, file_path)."""
    n = max(1, int(num_examples))
    threshold = float(dedup_threshold) if dedup_threshold and dedup_threshold > 0 else None

    try:
        backend = build_backend(
            kind=backend_kind, model_id=model_id,
            load_in_4bit=load_in_4bit,
            api_base_url=api_base_url, api_key=api_key or None,
        )
    except Exception as exc:
        return _error_block(f"Failed to build backend: {exc}"), "", "", None, {}

    dedup_model = _load_dedup_model(threshold) if threshold is not None else None
    dedup_priors: list = []
    dedup_drops = 0

    conversations = []
    failures = 0
    started = time.monotonic()

    current_topic = topic
    current_scenario = scenario
    current_style = style

    extras = _parse_extra_personas(extra_personas)
    base_personas = [(persona1, persona1_desc or ""), (persona2, persona2_desc or "")]
    full_personas = [p for p in base_personas + extras if p[0].strip()]

    # Build role_mapping if user picked a specific speaker for the gpt role
    role_mapping = None
    if train_speaker and train_speaker != "auto":
        persona_names = [n for n, _ in full_personas]
        if train_speaker in persona_names:
            role_mapping = build_role_mapping(persona_names, train_speaker=train_speaker)

    points = include_points.strip() if include_points else None

    for i in range(n):
        if enable_variation and i > 0 and n > 1:
            try:
                varied = generate_topic_variation(
                    persona1=persona1, persona1_desc=persona1_desc,
                    persona2=persona2, persona2_desc=persona2_desc,
                    initial_topic=topic, initial_scenario=scenario,
                    initial_style=style, backend=backend,
                )
                if varied:
                    current_topic = varied.get("topic") or topic
                    current_scenario = varied.get("scenario") or scenario
                    current_style = varied.get("style") or style
            except Exception:
                pass

        turns = generate_conversation(
            topic=current_topic,
            personas=full_personas,
            scenario=current_scenario, style=current_style,
            backend=backend,
            max_new_tokens=int(max_new_tokens),
            include_points=points,
            role_mapping=role_mapping,
        )
        if not turns:
            failures += 1
            continue

        if threshold is not None and dedup_model is not None:
            if _dedup_check(turns, dedup_model, dedup_priors, threshold):
                dedup_drops += 1
                continue

        conversations.append({
            "turns": turns,
            "topic": current_topic,
            "scenario": current_scenario,
            "style": current_style,
            "include_points": "",
        })

    if not conversations:
        return (
            _error_block(
                "No conversations were produced. Common causes: model not found, "
                "wrong API URL, server not running. Check the terminal."
            ),
            "", "", None, {},
        )

    elapsed = time.monotonic() - started

    fd, jsonl_path = tempfile.mkstemp(prefix="cdg-dataset-", suffix=".jsonl")
    os.close(fd)
    total_turns = write_jsonl(conversations, jsonl_path)

    metrics = _compute_metrics(jsonl_path) if len(conversations) > 1 else {}
    metrics_md = _format_metrics_card(metrics, dedup_drops=dedup_drops) if metrics else ""

    summary_bits = [
        f"<strong>{len(conversations)} conversation"
        f"{'s' if len(conversations) != 1 else ''}</strong>"
        f" · {total_turns} turns · {elapsed:.1f}s",
    ]
    if failures:
        summary_bits.append(f"{failures} generation failures")
    if dedup_drops:
        summary_bits.append(f"{dedup_drops} near-duplicates dropped")
    status = _success_block(" · ".join(summary_bits))

    preview = _format_preview(conversations, limit=3)
    return status, metrics_md, preview, jsonl_path, metrics


# ----------------------------- UI ---------------------------------------------


PAPER = "#f4efe5"
INK = "#16140f"
INK_SOFT = "#4a4538"
RULE = "#d8cfbb"
RULE_SOFT = "#e6dfcc"
ACCENT = "#c2410c"
ACCENT_SOFT = "#fde4d3"


def _theme():
    """A Gradio theme that erases the default blue and renders cleanly on cream paper."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50=ACCENT_SOFT, c100=ACCENT_SOFT, c200="#fbcdb1",
            c300="#f7a880", c400="#ee7c4a", c500=ACCENT,
            c600="#a8390a", c700="#8c2f08", c800="#6b2406",
            c900="#4d1a04", c950="#2c0f02",
        ),
        neutral_hue=gr.themes.Color(
            c50=PAPER, c100="#ece6d6", c200=RULE_SOFT, c300=RULE,
            c400="#bdb29a", c500=INK_SOFT, c600="#3d3a2d",
            c700="#2d2a20", c800="#1f1c14", c900=INK, c950="#0c0a07",
        ),
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace"],
    ).set(
        body_background_fill=PAPER,
        body_background_fill_dark=PAPER,
        body_text_color=INK,
        body_text_color_subdued=INK_SOFT,
        background_fill_primary=PAPER,
        background_fill_secondary=PAPER,
        background_fill_primary_dark=PAPER,
        background_fill_secondary_dark=PAPER,
        block_background_fill=PAPER,
        block_background_fill_dark=PAPER,
        block_border_width="0px",
        block_border_color=RULE,
        block_label_background_fill=PAPER,
        block_label_text_color=INK_SOFT,
        block_label_text_size="10px",
        block_label_text_weight="500",
        block_radius="0px",
        block_shadow="none",
        input_background_fill=PAPER,
        input_background_fill_focus=PAPER,
        input_background_fill_dark=PAPER,
        input_background_fill_hover=PAPER,
        input_border_color=RULE,
        input_border_color_focus=ACCENT,
        input_border_color_hover=INK_SOFT,
        input_border_width="0 0 1px 0",
        input_radius="0px",
        input_shadow="none",
        input_shadow_focus="none",
        button_primary_background_fill=INK,
        button_primary_background_fill_hover=ACCENT,
        button_primary_text_color=PAPER,
        button_primary_text_color_hover=PAPER,
        button_primary_border_color=INK,
        button_primary_border_color_hover=ACCENT,
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        button_secondary_background_fill=PAPER,
        button_secondary_background_fill_hover=RULE_SOFT,
        button_secondary_text_color=INK,
        button_secondary_border_color=RULE,
        slider_color=ACCENT,
        slider_color_dark=ACCENT,
        color_accent=ACCENT,
        color_accent_soft=ACCENT_SOFT,
        border_color_accent=ACCENT,
        border_color_accent_subdued=ACCENT_SOFT,
        border_color_primary=RULE,
        border_color_primary_dark=RULE,
        checkbox_background_color=PAPER,
        checkbox_background_color_focus=PAPER,
        checkbox_background_color_hover=PAPER,
        checkbox_background_color_selected=ACCENT,
        checkbox_border_color=RULE,
        checkbox_border_color_focus=ACCENT,
        checkbox_border_color_selected=ACCENT,
        checkbox_label_background_fill=PAPER,
        checkbox_label_background_fill_hover=PAPER,
        checkbox_label_background_fill_selected=PAPER,
        checkbox_label_text_color=INK,
        checkbox_label_text_color_selected=INK,
        checkbox_label_border_color=RULE,
        checkbox_label_border_width="0px",
        radio_circle=ACCENT,
        panel_background_fill=PAPER,
        panel_border_width="0px",
    )


def build_ui() -> gr.Blocks:
    default_backend = os.environ.get("CDG_BACKEND", "openai")
    default_base_url = os.environ.get("CDG_BASE_URL", "http://localhost:11434/v1")
    default_model_id = os.environ.get("CDG_MODEL_ID", "llama3.2:1b")

    with gr.Blocks(title="Conversation Dataset Generator", fill_height=True, fill_width=True) as demo:
        gr.HTML(
            """
            <div id="cdg-mast">
                <div class="title">Conversation <em>Dataset</em> Generator</div>
                <div class="meta">manual · sharegpt · v1</div>
            </div>
            """
        )

        with gr.Row(elem_id="cdg-grid", equal_height=False):
            # ----------------- LEFT: settings -------------
            with gr.Column(elem_classes=["cdg-pane", "left"], scale=0, min_width=360):
                gr.HTML(
                    '<div class="cdg-section"><div class="heading">'
                    '<span class="num">01</span><span class="name">Backend</span>'
                    '</div></div>'
                )
                with gr.Row():
                    backend_kind = gr.Radio(
                        ["hf", "openai"], value=default_backend, label="Backend", scale=1,
                    )
                    model_id = gr.Textbox(value=default_model_id, label="Model ID", scale=1)
                with gr.Row():
                    api_base_url = gr.Textbox(value=default_base_url, label="Base URL (openai)", scale=2)
                    api_key = gr.Textbox(value="", label="API key", type="password", scale=1)
                with gr.Row():
                    load_in_4bit = gr.Checkbox(value=False, label="4-bit (hf)", scale=1)
                    max_new_tokens = gr.Slider(
                        minimum=256, maximum=4096, value=1024, step=128,
                        label="Max new tokens", scale=2,
                    )

                gr.HTML(
                    '<div class="cdg-section"><div class="heading">'
                    '<span class="num">02</span><span class="name">Personas</span>'
                    '</div></div>'
                )
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=list(PRESETS.keys())[0],
                    label="Preset",
                    info="Pick a curated cast, write your own, or use a creative brief below.",
                )
                with gr.Row(elem_id="cdg-brief-row"):
                    brief = gr.Textbox(
                        value="", label="Or paste a creative brief",
                        placeholder='e.g. "Sherlock and Watson debate AI surveillance"',
                        scale=3,
                    )
                    brainstorm_btn = gr.Button(
                        "Brainstorm", elem_id="cdg-brainstorm-btn", scale=1,
                    )
                with gr.Row():
                    persona1 = gr.Textbox(value="Alice", label="Name 1", scale=1)
                    persona2 = gr.Textbox(value="Bob", label="Name 2", scale=1)
                with gr.Row():
                    persona1_desc = gr.Textbox(
                        value="A friendly engineer who loves explaining things",
                        label="Desc 1", lines=2, scale=1,
                    )
                    persona2_desc = gr.Textbox(
                        value="A curious student new to the topic",
                        label="Desc 2", lines=2, scale=1,
                    )
                extra_personas = gr.Textbox(
                    value="",
                    label="Add more (optional, one per line)",
                    info="Format: Name | Description",
                    placeholder="Carol | A skeptical scientist\nDave | An eager intern",
                    lines=3,
                )
                train_speaker = gr.Dropdown(
                    choices=["auto", "Alice", "Bob"],
                    value="auto",
                    label="Train speaker (which speaker = gpt role)",
                    info="The 'gpt' role is what your model learns to imitate. Auto: first = human, rest = gpt.",
                    allow_custom_value=True,
                )

                gr.HTML(
                    '<div class="cdg-section"><div class="heading">'
                    '<span class="num">03</span><span class="name">Scene</span>'
                    '</div></div>'
                )
                topic = gr.Textbox(value="how transformers work", label="Topic")
                with gr.Row():
                    scenario = gr.Textbox(value="a quiet coffee shop", label="Scenario", scale=1)
                    style = gr.Textbox(value="Casual and curious", label="Style", scale=1)
                include_points = gr.Textbox(
                    value="", label="Must-cover points (optional)",
                    info="Comma-separated points the dialogue must address.",
                    placeholder="e.g. attention mechanism, embeddings, training cost",
                )

                gr.HTML(
                    '<div class="cdg-section"><div class="heading">'
                    '<span class="num">04</span><span class="name">Batch</span>'
                    '</div></div>'
                )
                num_examples = gr.Slider(
                    minimum=1, maximum=50, value=1, step=1,
                    label="Number of conversations",
                )
                with gr.Row():
                    enable_variation = gr.Checkbox(
                        value=True, label="Vary each", scale=1,
                    )
                    dedup_threshold = gr.Slider(
                        minimum=0.0, maximum=0.99, value=0.0, step=0.01,
                        label="Dedup threshold (0 = off)", scale=2,
                    )

                generate_btn = gr.Button("Generate dataset", elem_id="cdg-generate-btn")

            # ----------------- RIGHT: dashboard zones -------------
            with gr.Column(elem_classes=["cdg-pane"], elem_id="cdg-right", scale=1):
                gr.HTML(
                    '<div class="cdg-section" style="margin-bottom:6px"><div class="heading">'
                    '<span class="num">A</span><span class="name">Run status</span>'
                    '</div></div>'
                )
                status_md = gr.Markdown(
                    value='<div class="cdg-loading">Awaiting generate.</div>',
                    elem_id="cdg-status",
                )

                gr.HTML(
                    '<div class="cdg-section" style="margin-top:14px;margin-bottom:6px"><div class="heading">'
                    '<span class="num">B</span><span class="name">Diversity metrics</span>'
                    '</div></div>'
                )
                metrics_md = gr.Markdown(
                    value='<em style="font-family:var(--serif);color:var(--ink-soft)">'
                          'Computed after each run.</em>',
                    elem_id="cdg-metrics",
                )

                metrics_state = gr.State(value={})
                with gr.Row(elem_id="cdg-fix-row"):
                    fix_btn = gr.Button(
                        "Auto-fix issues",
                        elem_id="cdg-fix-btn",
                    )
                    fix_status = gr.Markdown(
                        value="", elem_id="cdg-fix-status",
                    )

                gr.HTML(
                    '<div class="cdg-section" style="margin-top:14px;margin-bottom:6px"><div class="heading">'
                    '<span class="num">C</span><span class="name">Dataset</span>'
                    '</div></div>'
                )
                file_out = gr.File(
                    label="ShareGPT JSONL — download for training",
                    interactive=False,
                )

                gr.HTML(
                    '<div class="cdg-section" style="margin-top:14px;margin-bottom:6px"><div class="heading">'
                    '<span class="num">D</span><span class="name">Preview</span>'
                    '</div></div>'
                )
                preview_md = gr.Markdown(
                    value='<em style="font-family:var(--serif);color:var(--ink-soft)">'
                          'First three conversations render here. Long batches scroll within this panel.</em>',
                    elem_id="cdg-preview",
                )

        inputs = [
            backend_kind, model_id, api_base_url, api_key, load_in_4bit,
            persona1, persona1_desc, persona2, persona2_desc,
            topic, scenario, style, max_new_tokens,
            num_examples, enable_variation, dedup_threshold,
            extra_personas, train_speaker, include_points,
        ]
        outputs = [status_md, metrics_md, preview_md, file_out, metrics_state]

        # Preset → fill form
        preset_dropdown.change(
            _apply_preset,
            inputs=[preset_dropdown],
            outputs=[persona1, persona1_desc, persona2, persona2_desc,
                     extra_personas, topic, scenario, style],
        )

        # Brainstorm button → LLM populates persona/scene fields from brief
        brainstorm_btn.click(
            lambda: '<em style="color:var(--ink-soft);font-family:var(--serif)">Asking the model to brainstorm…</em>',
            outputs=[fix_status],
        ).then(
            brainstorm_handler,
            inputs=[backend_kind, model_id, api_base_url, api_key, load_in_4bit, brief],
            outputs=[persona1, persona1_desc, persona2, persona2_desc,
                     topic, scenario, style, fix_status],
        )

        # Auto-fix issues → run every applicable fix based on latest metrics
        fix_btn.click(
            lambda: '<em style="color:var(--ink-soft);font-family:var(--serif)">Diagnosing issues and applying fixes…</em>',
            outputs=[fix_status],
        ).then(
            auto_fix_handler,
            inputs=[
                backend_kind, model_id, api_base_url, api_key, load_in_4bit,
                persona1, persona1_desc, persona2, persona2_desc, extra_personas,
                topic, scenario, style, max_new_tokens, enable_variation,
                dedup_threshold, metrics_state,
            ],
            outputs=[
                persona1_desc, persona2_desc, extra_personas,
                topic, scenario, style,
                max_new_tokens, enable_variation,
                fix_status,
            ],
        )

        # When persona names change, refresh the train_speaker dropdown choices
        def _refresh_train_speaker_choices(p1, p2, extras):
            extras_parsed = _parse_extra_personas(extras)
            names = ["auto"] + [n for n in [p1, p2] if n.strip()] + [n for n, _ in extras_parsed]
            return gr.update(choices=names, value="auto" if "auto" in names else names[0])
        for component in (persona1, persona2, extra_personas):
            component.change(
                _refresh_train_speaker_choices,
                inputs=[persona1, persona2, extra_personas],
                outputs=[train_speaker],
            )

        generate_btn.click(
            lambda: (
                '<div class="cdg-loading">Generating dataset… this can take a while for large batches.</div>',
                "", "", None, {},
            ),
            outputs=outputs,
        ).then(generate_handler, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=_theme(), css=CUSTOM_CSS)
