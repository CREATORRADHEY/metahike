from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import subprocess
import json

from core.tasks import TASKS
from core.env import SupportEnv
from core.types import Action

app = FastAPI(title="Customer Support AI Training Environment", docs_url="/")

class GraderRequest(BaseModel):
    task_id: str
    ticket_id: str
    history: List[Dict[str, Any]]

class ResetRequest(BaseModel):
    task_id: str = "task_categorize_easy"

# Global environment instance for simple evaluation pings
global_env = SupportEnv()

@app.post("/reset")
def env_reset(req: ResetRequest):
    try:
        obs = global_env.reset(req.task_id)
        return {"observation": obs.dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def env_step(action: Action):
    try:
        obs, reward, done, info = global_env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def env_state():
    return global_env.state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": task_id, "description": task.description}
            for task_id, task in TASKS.items()
        ],
        "action_schema": {
            "type": "dict",
            "properties": {
                "action_type": "string (e.g., submit, lookup_policy)",
                "value": "dict (e.g., {'category': 'Billing', 'order_id': '123'})"
            }
        }
    }

@app.post("/grader")
def run_grader(req: GraderRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = TASKS[req.task_id]
    
    # Find the specific instance by ticket_id to grade against
    instance = next((inst for inst in getattr(task, "instances", []) if inst.ticket_id == req.ticket_id), None)
    if not instance:
        raise HTTPException(status_code=404, detail="Ticket ID not found for this task")
        
    score, reason, done = task.evaluate(instance, req.history)
    return {
        "score": score,
        "reason": reason,
        "done": done
    }

@app.post("/baseline")
def run_baseline():
    try:
        # Run the inference script inside a subprocess
        # We capture stdout and parse it
        result = subprocess.run(["python", "inference.py"], capture_output=True, text=True, check=True)
        # Assuming baseline.py prints a JSON with scores at the very end
        lines = result.stdout.strip().split('\n')
        # find the last valid json line
        scores = {}
        for line in reversed(lines):
            try:
                scores = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
                
        return {"baseline_scores": scores, "logs": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {e.stderr}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
