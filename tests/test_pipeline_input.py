import pytest

from src.runtime.pipeline import ProcessingPipeline


def test_pipeline_rejects_empty_frame_before_processing():
    pipeline = object.__new__(ProcessingPipeline)
    with pytest.raises(ValueError, match="empty or invalid"):
        pipeline.process(None)
