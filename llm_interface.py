# llm_interface.py
from typing import Dict

# Імітація виклику LLM (у реальному випадку - виклик до OpenAI, Mistral тощо)
def parse_task_description(prompt: str) -> Dict:
    # Проста евристика для демо. Заміни на запит до LLM
    schema = {
        "task_type": "classification" if "класифікація" in prompt else "regression",
        "data_type": "tabular" if any(word in prompt.lower() for word in ["табличн", "csv", "файл", "дані", "впливають", "стовпець"]) else "text",
        "target": "sentiment" if "відгук" in prompt else "target",
        "preferences": {
            "speed": "high" if "швидк" in prompt else "normal",
            "interpretability": "yes" if "інтерпрет" in prompt else "no"
        }
    }
    return schema


def generate_pipeline_code(schema: Dict) -> str:
    if schema["data_type"] == "tabular":
        return f'''from autogluon.tabular import TabularPredictor
import pandas as pd

train_data = pd.read_csv("input_data.csv")
predictor = TabularPredictor(label="{schema['target']}", problem_type="{schema['task_type']}")\
    .fit(train_data=train_data, time_limit=300)
leaderboard = predictor.leaderboard(silent=True)
leaderboard.to_csv("leaderboard.csv", index=False)
model = predictor'''
    else:
        return "raise NotImplementedError('Підтримуються лише табличні дані')"
