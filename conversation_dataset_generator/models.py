"""Model and tokenizer loading, pipeline creation."""

import logging
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_model_and_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = False,
) -> tuple:
    """Load tokenizer and model, return (pipeline, tokenizer).

    Args:
        model_id: HuggingFace model identifier.
        load_in_4bit: Whether to use 4-bit NF4 quantization.

    Returns:
        Tuple of (text-generation pipeline, tokenizer).
    """
    start = time.monotonic()

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        logger.info("Loading model with 4-bit quantization (NF4): %s", model_id)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.info("Loading model with default precision: %s", model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            device_map="auto",
            trust_remote_code=True,
        )

    logger.info("Creating text-generation pipeline...")
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    elapsed = time.monotonic() - start
    logger.info("Model loaded in %.2fs (4-bit: %s)", elapsed, load_in_4bit)

    return text_generator, tokenizer
