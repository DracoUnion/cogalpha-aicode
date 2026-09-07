"""CogAlpha: Cognitive Alpha Mining via LLM-Driven Code-Based Evolution.

A faithful, re-implementable version of the framework described in
`paper.md`, built on the public prompt templates in `prompts/`.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
