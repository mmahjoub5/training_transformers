import os 
from typing import List, Dict, Any, Optional
import json

def read_prompts(path: str) -> List[str]:
    """
    Supports:
      - .txt (one prompt per line; blank lines ignored)
      - .jsonl (each line JSON with key 'prompt' or 'question')
      - .json  (list of strings OR list of {"prompt": "..."} dicts)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    prompts: List[str] = []
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    prompts.append(s)

    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = obj.get("prompt") or obj.get("question")
                if not p:
                    raise ValueError("jsonl lines must contain 'prompt' or 'question'")
                prompts.append(str(p).strip())

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict):
                    p = item.get("prompt") or item.get("question")
                    if not p:
                        raise ValueError("json list dict items must contain 'prompt' or 'question'")
                    prompts.append(str(p).strip())
                else:
                    raise ValueError("json list items must be strings or dicts")
        else:
            raise ValueError(".json must contain a list")
    else:
        raise ValueError("Unsupported file type. Use .txt, .jsonl, or .json")

    if not prompts:
        raise ValueError("No prompts found.")
    return prompts