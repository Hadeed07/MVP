from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image


class qwen:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    ):

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        def predict(self, img):
            messages = [
                {"role": "system", "content": "You are an OCR assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {
                            "type": "text",
                            "text": """
                    Extract every visible word exactly as written.

                    Do not translate.
                    Do not summarize.
                    Do not infer missing text.

                    After extracting the text, identify:

                    Title:
                    Author:
                    Language:
                    """,
                        },
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[text],
                images=[img],
                return_tensors="pt",
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            generated_ids_trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response

        def predict_from_path(self, image_path):
            img = Image.open(image_path)
            return self.predict(img)
        
