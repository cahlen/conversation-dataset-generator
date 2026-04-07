#!/usr/bin/env python
# coding=utf-8
"""Evaluate generated conversation datasets."""

import argparse
import logging
import sys

from conversation_dataset_generator.evaluation import run_evaluation, format_report, format_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated conversation dataset.")
    parser.add_argument("input", help="Path to JSONL file to evaluate.")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-conversation breakdown.")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding-based metrics (faster, no model download).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_name = None if args.no_embeddings else "sentence-transformers/all-MiniLM-L6-v2"
    results = run_evaluation(args.input, model_name=model_name)

    if args.format == "json":
        print(format_json(results))
    else:
        print(format_report(results))


if __name__ == "__main__":
    main()
