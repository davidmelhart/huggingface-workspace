import os
import csv
from datetime import datetime
from models.qwen import Qwen2_5_VL

if __name__ == "__main__":
    with Qwen2_5_VL() as model:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/data/test1.jpg"},
                    {"type": "image", "image": "/data/test2.jpg"},
                    {"type": "text", "text": "What is the difference between these images?"},
                ],
            },
        ]
        output = model.infer(messages)

        result = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model.model_name,
            'input': messages,
            'output': output
        }

        log_file = os.path.join('output', f'logs.csv')
        with open(log_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if not os.path.isfile(log_file):
                writer.writeheader()
                results_exist = True
            writer.writerow(result)

        print(f'\n{model.model_name}:\n\t"{output}"\n')
    print('\n>>> Finished!')
