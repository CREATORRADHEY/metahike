---
title: Metahike Support Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Support Simulator: AI Training Environment

## What is this environment?

The Customer Support Simulator is a minimal AI training environment designed to teach agents how to handle real-world customer support tickets. Instead of a game, the agent interacts with support tickets (e.g., billing disputes, technical issues, refunds) via a standard `reset()`, `step()`, `state()` RL loop. The environment grades the agent's actions on a deterministic scale (0.0 to 1.0) and offers partial rewards to facilitate learning along the episode trail.

## Why is it useful?

It bridges the gap between traditional Reinforcement Learning (RL) benchmarks and practical AI agent workflows. By training against this arena, researchers can evaluate how well Large Language Models and RL agents understand policies, extract specific entities, and respond appropriately under real business conditions without requiring access to a live production database.

## Observation and Action Format

This environment operates using structured Pydantic models (JSON schema).

**Observation:**
```json
{
  "ticket_text": "I was charged twice for my subscription this month, please fix.",
  "history": [{"action_type": "think", "value": "..."}],
  "metadata": {"ticket_id": "T001"}
}
```

**Action:**
```json
{
  "action_type": "submit",
  "value": {
    "category": "Billing",
    "order_id": "none",
    "draft": "We have refunded the duplicate charge."
  }
}
```

## The 3 Tasks

The arena provides 3 tasks of increasing difficulty levels:

1. **`task_categorize_easy` (Difficulty: Easy)**
   * **Goal:** Simply route the incoming ticket into one of three categories: "Billing", "Tech Support", or "Refund".
   * **Grading:** 1.0 for the correct category upon submission.

2. **`task_extract_medium` (Difficulty: Medium)**
   * **Goal:** Categorize the ticket accurately AND extract the `order_id` present in the user's message.
   * **Grading:** Reward is split. 0.5 for a correct category, 0.5 for successfully extracting the correct Order ID. 

3. **`task_draft_hard` (Difficulty: Hard)**
   * **Goal:** A multi-step task where the agent must evaluate the ticket, perform a `lookup_policy` action for the relevant rule (e.g., Refund vs Billing rules), and finally `submit` a drafted email response that complies with the underlying policy constraints.
   * **Grading:** Granular partial rewards (e.g., 0.1 for rule lookup, 0.8 for compliance with rule within the final response).

## Installation & Running

This environment can be deployed directly via Docker + Hugging Face Spaces.

**Build and Run with Docker:**
```bash
docker build -t support-env .
docker run -p 7860:7860 support-env
```
The FastAPI wrapper will expose the environment on `http://localhost:7860`.
Endpoints:
* `GET /tasks`
* `POST /grader`
* `POST /baseline` (runs inference.py)

## How to Run Inference

The inference script is a zero-shot, prompt-based loop leveraging `gpt-4o-mini` to attempt the tasks.

```bash
export OPENAI_API_KEY="sk-..."
python inference.py
```

## Baseline Scores

When executed with `gpt-4o-mini`, standard deterministic baseline scores are roughly:
* **`task_categorize_easy`**: 1.0
* **`task_extract_medium`**: 1.0
* **`task_draft_hard`**: ~0.9 (depending on exact syntax compliance in the response draft)

```json
{"task_categorize_easy": 1.0, "task_extract_medium": 1.0, "task_draft_hard": 0.9}
```
