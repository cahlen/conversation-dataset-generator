"""Tests for the ChatBackend protocol and its implementations."""

import pytest
from unittest.mock import MagicMock


class TestImports:
    def test_chat_backend_protocol_importable(self):
        from conversation_dataset_generator.backend import ChatBackend
        assert ChatBackend is not None
