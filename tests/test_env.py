import pytest
from core.env import SupportEnv
from core.types import Action
from core.tasks import TASKS

def test_easy_task():
    env = SupportEnv()
    obs = env.reset("task_categorize_easy")
    assert "ticket_id" in obs.metadata
    
    # Send a think action
    obs, reward, done, info = env.step(Action(action_type="think", value={}))
    assert reward.score == 0.1
    assert not done
    
    # Send correct category
    correct_cat = env.instance.metadata["topic"]
    obs, reward, done, info = env.step(Action(action_type="submit", value={"category": correct_cat}))
    assert reward.score == 1.0 # The score should accumulate to 1.1 if not bound, but env.step bounds it
    assert info["raw_score"] == 1.1
    assert done
    assert 0.0 <= reward.score <= 1.0

def test_medium_task():
    env = SupportEnv()
    obs = env.reset("task_extract_medium")
    
    # Correct order ID, wrong category
    correct_order = env.instance.metadata["order_id"]
    obs, reward, done, info = env.step(Action(action_type="submit", value={"category": "Wrong", "order_id": correct_order}))
    assert reward.score == 0.5
    assert done

def test_hard_task():
    env = SupportEnv()
    obs = env.reset("task_draft_hard")
    
    topic = env.instance.metadata["topic"]
    
    # Lookup policy
    obs, reward, done, info = env.step(Action(action_type="lookup_policy", value={"topic": topic}))
    assert reward.score == 0.1
    assert not done
    
    # Submit draft
    # Just mock a bad draft
    obs, reward, done, info = env.step(Action(action_type="submit", value={"draft": "fake draft"}))
    assert reward.score == pytest.approx(0.3)  # 0.1 from lookup + 0.2 from submit (because looked up)
    assert done

def test_graders_bounds():
    env = SupportEnv()
    # Ensure no task returns score outside [0, 1] for typical correct paths
    for task_id in TASKS:
        env.reset(task_id)
        # Check defaults
        assert env.score == 0.0
