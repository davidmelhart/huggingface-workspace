import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.llm_base import VisionLanguageModelPrototype


class Qwen2_5_VL(VisionLanguageModelPrototype):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        super().__init__(model_name)

    def load(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"{self.model_name} model and processor loaded successfully.")

    def infer(self, messages, image_files=[], max_new_tokens=2048, temperature=0.0):
        """
        Perform inference using the loaded model.

        Args:
            messages (list): List of message dictionaries containing image paths and text.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file:///path/to/image1.jpg"},
                        {"type": "image", "image": "file:///path/to/image2.jpg"},
                        {"type": "text", "text": "Identify the similarities between these images."},
                    ],
                }
            ]
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Temperature for controlling randomness in generation.

        Returns:
            str: Generated text output.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and/or processor not loaded. Call load() first.")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False if temperature == 0 else True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
