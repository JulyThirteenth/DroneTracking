from dataclasses import dataclass
from typing import Literal
import yaml, os
from pydantic import SecretStr

@dataclass
class ModelConfig:
    id: str
    provider: Literal["openai", "google"]
    model_name: str
    api_key_env: str
    temperature: float = 0.7
    base_url: str | None = None

    @property
    def api_key(self) -> SecretStr:
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(f"环境变量 {self.api_key_env} 未设置")
        return SecretStr(key)

@dataclass
class Task:
    id: str
    type: Literal["pos", "desp", "img"]
    difficulty: Literal["same_room", "cross_room"]
    message: str

def load_models(yaml_path: str) -> dict[str, ModelConfig]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    models = [ModelConfig(**m) for m in data["models"]]
    return {m.id: m for m in models}

def load_tasks(yaml_path: str) -> list[Task]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [Task(**t) for t in data["tasks"]]
