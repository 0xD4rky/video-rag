import os
import sys
import importlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))



def test_query_expander():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available")
    from query_expander import QueryExpander
    expander = QueryExpander(gemini_api_key=None)
    expanded = expander.expand("cat running")
    assert isinstance(expanded, str) and len(expanded) > 0


def test_extract_scenes():
    from scene import extract_scenes
    scenes, fps = extract_scenes("data/test_1.mp4", scene_duration=2, fps=1)
    assert len(scenes) > 0
    assert fps > 0


def test_video_agent(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available")
    from video_agent import VideoAgent
    agent = VideoAgent(use_serpapi=False)
    output = agent.run("data/test_1.mp4", "cat", output_dir=str(tmp_path))
    assert output is not None
    assert os.path.exists(output)

