"""Gradio web interface for conversation generation.

Run with:
    python webapp.py

Single-page UI for manual mode (two personas + topic/scenario/style). Backend
is selectable: 'hf' for local transformers, 'openai' for any OpenAI-compatible
HTTP server (LM Studio, Ollama, vLLM, OpenAI itself).
"""

from __future__ import annotations

import gradio as gr

from conversation_dataset_generator.cli import build_backend
from conversation_dataset_generator.generation import generate_conversation


CUSTOM_CSS = """
:root {
    --cdg-radius: 14px;
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
}
#cdg-hero {
    padding: 28px 32px;
    border-radius: var(--cdg-radius);
    background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 45%, #0ea5e9 100%);
    color: #ffffff;
    margin-bottom: 18px;
}
#cdg-hero h1 {
    margin: 0 0 6px 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
}
#cdg-hero p {
    margin: 0;
    opacity: 0.85;
    font-size: 14px;
    color: #ffffff;
}
.cdg-card {
    border: 1px solid var(--border-color-primary);
    border-radius: var(--cdg-radius);
    padding: 16px 18px !important;
}
.cdg-section-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    opacity: 0.65;
    margin: 0 0 10px 0;
}
#cdg-output {
    min-height: 420px;
    padding: 20px 24px !important;
    background: var(--background-fill-secondary);
    border-radius: var(--cdg-radius);
    border: 1px solid var(--border-color-primary);
    font-size: 14.5px;
    line-height: 1.7;
}
#cdg-generate-btn {
    border-radius: 999px !important;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.cdg-footer {
    margin-top: 18px;
    text-align: center;
    font-size: 12px;
    opacity: 0.55;
}
"""


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
) -> str:
    """Build a backend, run one conversation, return formatted markdown."""
    try:
        backend = build_backend(
            kind=backend_kind,
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            api_base_url=api_base_url,
            api_key=api_key or None,
        )
    except Exception as exc:
        return f"**Failed to build backend:** {exc}"

    turns = generate_conversation(
        topic=topic,
        persona1=persona1, persona1_desc=persona1_desc,
        persona2=persona2, persona2_desc=persona2_desc,
        scenario=scenario, style=style,
        backend=backend,
        max_new_tokens=int(max_new_tokens),
    )

    if not turns:
        return "**Generation failed.** The backend returned no usable text — check server logs."

    return "\n\n".join(
        f"**{turn.get('speaker_name', '?')}:** {turn['value']}" for turn in turns
    )


def _theme():
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="sky",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Conversation Dataset Generator") as demo:
        gr.HTML(
            """
            <div id="cdg-hero">
                <h1>Conversation Dataset Generator</h1>
                <p>Generate synthetic two-speaker dialogue against a local transformers model
                or any OpenAI-compatible server (LM Studio, Ollama, OpenAI).</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=420):
                with gr.Group(elem_classes=["cdg-card"]):
                    gr.HTML('<p class="cdg-section-label">Backend</p>')
                    backend_kind = gr.Radio(
                        ["hf", "openai"], value="openai", label="Inference backend",
                        info="`hf` runs locally with transformers; `openai` hits any OpenAI-compatible server.",
                    )
                    model_id = gr.Textbox(
                        value="llama3.2:1b", label="Model ID",
                        info="HF id for the hf backend; the name your server expects for openai.",
                    )
                    with gr.Accordion("OpenAI server settings", open=True):
                        api_base_url = gr.Textbox(
                            value="http://localhost:11434/v1", label="API base URL",
                            info="LM Studio: http://localhost:1234/v1 — Ollama: http://localhost:11434/v1",
                        )
                        api_key = gr.Textbox(
                            value="", label="API key (optional)", type="password",
                            info="Falls back to OPENAI_API_KEY env, then 'not-needed'.",
                        )
                    with gr.Accordion("Advanced", open=False):
                        load_in_4bit = gr.Checkbox(value=False, label="Load in 4-bit (hf only)")
                        max_new_tokens = gr.Slider(
                            minimum=256, maximum=4096, value=1024, step=128,
                            label="Max new tokens",
                        )

                with gr.Group(elem_classes=["cdg-card"]):
                    gr.HTML('<p class="cdg-section-label">Personas</p>')
                    with gr.Row():
                        persona1 = gr.Textbox(value="Alice", label="Persona 1 — name", scale=1)
                        persona2 = gr.Textbox(value="Bob", label="Persona 2 — name", scale=1)
                    persona1_desc = gr.Textbox(
                        value="A friendly engineer who loves explaining things",
                        label="Persona 1 — description", lines=2,
                    )
                    persona2_desc = gr.Textbox(
                        value="A curious student new to the topic",
                        label="Persona 2 — description", lines=2,
                    )

                with gr.Group(elem_classes=["cdg-card"]):
                    gr.HTML('<p class="cdg-section-label">Scene</p>')
                    topic = gr.Textbox(value="how transformers work", label="Topic")
                    with gr.Row():
                        scenario = gr.Textbox(value="a quiet coffee shop", label="Scenario", scale=1)
                        style = gr.Textbox(value="Casual and curious", label="Style", scale=1)

                generate_btn = gr.Button(
                    "Generate conversation", variant="primary",
                    elem_id="cdg-generate-btn", size="lg",
                )

            with gr.Column(scale=7, min_width=480):
                with gr.Group(elem_classes=["cdg-card"]):
                    gr.HTML('<p class="cdg-section-label">Output</p>')
                    output = gr.Markdown(
                        value="*Generated conversation will appear here.*",
                        elem_id="cdg-output",
                    )

                gr.Examples(
                    label="Try a preset",
                    examples=[
                        [
                            "openai", "llama3.2:1b",
                            "http://localhost:11434/v1", "", False,
                            "Sherlock", "A coldly logical Victorian detective",
                            "Watson", "A loyal, slightly bewildered army doctor",
                            "the ethics of AI surveillance", "221B Baker Street",
                            "Tense and dramatic", 1024,
                        ],
                        [
                            "openai", "llama3.2:1b",
                            "http://localhost:11434/v1", "", False,
                            "Maya", "An enthusiastic chef obsessed with umami",
                            "Diego", "A skeptical food scientist",
                            "whether umami is overrated", "a steamy professional kitchen",
                            "Spirited and friendly", 1024,
                        ],
                        [
                            "openai", "llama3.2:1b",
                            "http://localhost:11434/v1", "", False,
                            "Ada", "A pragmatic senior engineer",
                            "Lin", "An eager intern asking great questions",
                            "why monoliths beat microservices for small teams",
                            "a code review session", "Direct and educational", 1024,
                        ],
                    ],
                    inputs=[
                        backend_kind, model_id, api_base_url, api_key, load_in_4bit,
                        persona1, persona1_desc, persona2, persona2_desc,
                        topic, scenario, style, max_new_tokens,
                    ],
                )

        gr.HTML(
            '<div class="cdg-footer">conversation-dataset-generator · '
            'manual mode · point at any OpenAI-compatible server to skip the GPU</div>'
        )

        generate_btn.click(
            generate_handler,
            inputs=[
                backend_kind, model_id, api_base_url, api_key, load_in_4bit,
                persona1, persona1_desc, persona2, persona2_desc,
                topic, scenario, style, max_new_tokens,
            ],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=_theme(), css=CUSTOM_CSS)
