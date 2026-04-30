"""Minimal Gradio web interface for conversation generation.

Run with:
    python webapp.py

Single page, manual mode (two personas + topic/scenario/style). Backend is
selectable: 'hf' for local transformers, 'openai' for any OpenAI-compatible
HTTP server (LM Studio, Ollama, vLLM, OpenAI itself).
"""

from __future__ import annotations

import gradio as gr

from conversation_dataset_generator.cli import build_backend
from conversation_dataset_generator.generation import generate_conversation


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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Conversation Dataset Generator") as demo:
        gr.Markdown(
            "# Conversation Dataset Generator\n"
            "Generate synthetic two-speaker dialogue. Point at a local "
            "transformers model (`hf`) or any OpenAI-compatible server (`openai`)."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Backend")
                backend_kind = gr.Radio(
                    ["hf", "openai"], value="openai", label="Backend",
                )
                model_id = gr.Textbox(
                    value="llama3.2:1b",
                    label="Model ID",
                    info="HF id for hf backend; whatever name your server expects for openai.",
                )
                api_base_url = gr.Textbox(
                    value="http://localhost:11434/v1",
                    label="API base URL (openai backend)",
                    info="LM Studio: http://localhost:1234/v1 — Ollama: http://localhost:11434/v1",
                )
                api_key = gr.Textbox(
                    value="", label="API key (optional)", type="password",
                    info="Falls back to OPENAI_API_KEY env, then 'not-needed'.",
                )
                load_in_4bit = gr.Checkbox(value=False, label="Load in 4-bit (hf only)")
                max_new_tokens = gr.Slider(
                    minimum=256, maximum=4096, value=1024, step=256,
                    label="Max new tokens",
                )

            with gr.Column():
                gr.Markdown("### Personas")
                persona1 = gr.Textbox(value="Alice", label="Persona 1 name")
                persona1_desc = gr.Textbox(
                    value="A friendly engineer who loves explaining things",
                    label="Persona 1 description",
                )
                persona2 = gr.Textbox(value="Bob", label="Persona 2 name")
                persona2_desc = gr.Textbox(
                    value="A curious student new to the topic",
                    label="Persona 2 description",
                )

            with gr.Column():
                gr.Markdown("### Scene")
                topic = gr.Textbox(value="how transformers work", label="Topic")
                scenario = gr.Textbox(value="a quiet coffee shop", label="Scenario")
                style = gr.Textbox(value="Casual and curious", label="Style")

        generate_btn = gr.Button("Generate", variant="primary")
        output = gr.Markdown(label="Generated conversation")

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
    build_ui().launch()
