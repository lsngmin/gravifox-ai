"""Detection package entry point."""

from . import gf1

# Friendly re-exports for default pipeline
SCRFDDetector = gf1.SCRFDDetector
run_video = gf1.run_video
run_media = gf1.run_media
config = gf1.config
pipeline = gf1.pipeline

__all__ = [
    "gf1",
    "SCRFDDetector",
    "run_video",
    "run_media",
    "config",
    "pipeline",
]
