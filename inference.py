import os
import json
import openai
from core.env import SupportEnv
from core.types import Action
from core.tasks import TASKS

def get_client():
    # Priority 1: Validator's proxy variables
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    
    # Priority 2: Fallback to standard OpenAI variable
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None
        
    return openai.OpenAI(api_key=api_key, base_url=base_url)

def run_task(env: SupportEnv, task_id: str, model="gpt-4o-mini") -> float:
    print(f"[START] task={task_id}", flush=True)
    client = get_client()
    if not client:
        print(f"Error: API_KEY/OPENAI_API_KEY not set. Cannot run task {task_id}.", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0
        
    obs = env.reset(task_id)
    done = False
    
    # Simple prompt describing the environment action space
    system_prompt = f"""
    You are an AI support agent. The current task is: {env.task.description}
    You must output a single JSON object representing your action.
    Action format:
    {{
        "action_type": "...",
        "value": {{...}}
    }}
    Possible action_types: 'lookup_policy', 'submit'.
    If the task requires extracting an 'order_id', include it in 'value'.
    If the task requires drafting a response, include it in 'value' as 'draft'.
    If the task requires categorization, include 'category' in 'value' (Options: Billing, Tech Support, Refund).
    If you need to lookup a policy, action_type='lookup_policy', value={{'topic': '...'}}
    ALWAYS REPLY WITH VALID JSON ONLY.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    step_count = 0
    while not done and step_count < 5:
        # Convert observation to message
        user_msg = f"TICKET TEXT: {obs.ticket_text}\nHISTORY: {json.dumps(obs.history)}\nMETADATA: {json.dumps(obs.metadata)}"
        messages.append({"role": "user", "content": user_msg})
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            action_json = json.loads(response.choices[0].message.content)
            
            # Record assistant msg
            messages.append({"role": "assistant", "content": json.dumps(action_json)})
            
            # Execute in env
            action = Action(**action_json)
            obs, reward, done, info = env.step(action)
            print(f"[STEP] step={step_count+1} reward={reward.score}", flush=True)
            
        except Exception as e:
            print(f"Error in step: {e}", flush=True)
            break
            
        step_count += 1
        
    print(f"[END] task={task_id} score={env.score} steps={step_count}", flush=True)
    return env.score

def main():
    env = SupportEnv()
    scores = {}
    
    if not os.environ.get("API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: No API key found (API_KEY or OPENAI_API_KEY). Baseline will report 0.0 for all tasks.", flush=True)

    for task_id in TASKS.keys():
        print(f"Running baseline for {task_id}...", flush=True)
        try:
            score = run_task(env, task_id)
            scores[task_id] = score
            print(f"Finished {task_id} with score: {score}", flush=True)
        except Exception as e:
            scores[task_id] = 0.0
            print(f"Failed {task_id}: {e}", flush=True)
            
    print("\n--- FINAL SCORES ---", flush=True)
    print(json.dumps(scores), flush=True)

if __name__ == "__main__":
    main()
