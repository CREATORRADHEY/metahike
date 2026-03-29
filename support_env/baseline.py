import os
import json
import openai
from core.env import SupportEnv
from core.types import Action
from core.tasks import TASKS

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_task(env: SupportEnv, task_id: str, model="gpt-4o-mini") -> float:
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
            
        except Exception as e:
            print(f"Error in step: {e}")
            break
            
        step_count += 1
        
    return env.score

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Cannot run baseline.")
        return

    env = SupportEnv()
    scores = {}
    
    for task_id in TASKS.keys():
        print(f"Running baseline for {task_id}...")
        try:
            score = run_task(env, task_id)
            scores[task_id] = score
            print(f"Finished {task_id} with score: {score}")
        except Exception as e:
            scores[task_id] = 0.0
            print(f"Failed {task_id}: {e}")
            
    print("\n--- FINAL SCORES ---")
    print(json.dumps(scores))

if __name__ == "__main__":
    main()
