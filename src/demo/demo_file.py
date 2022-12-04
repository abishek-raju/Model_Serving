import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml","setup.py"],
    pythonpath=True,
    dotenv=True,
)
from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

from src import utils

import numpy as np

from torchvision import transforms

import boto3
from os.path import exists


log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    file_exists = exists("model.script.pt")

    if not file_exists:
        log.info("Starting to download model")
        client = boto3.client('s3', 
                        aws_access_key_id="AKIA2AORTCZ66UZKRWJ3", 
                        aws_secret_access_key="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", 
                        region_name="ap-south-1"
                        )

        client.download_file(
        Bucket="script-models",
        Key = "model.script.pt",
        Filename = "model.script.pt"
        )
    else:
        log.info("Model already downloaded")

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{'model.script.pt'}>")
    model = torch.jit.load("model.script.pt")

    log.info(f"Loaded Model: {model}")
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def recognize_digit(image):
        if image is None:
            return None
        # print("image",image.shape)
        # tensor_image = torch.tensor(image, dtype=torch.float32)
        # image = image.convert("RGB")
        tensor_image = transforms.ToTensor()(image)
        # tensor_image = transforms.Resize((32,32))(tensor_image)
        tensor_image = tensor_image.unsqueeze(0)
        # print("tensor_image",tensor_image.shape)
        preds = model.forward_jit(tensor_image)
        # print(preds)
        preds = preds.tolist()
        return {labels[i]: preds[i] for i in range(10)}

    # im = gr.Image(shape=(28, 28), image_mode="L", invert_colors=True, source="canvas")
    im = gr.Image(type="numpy",shape=(32,32))

    demo = gr.Interface(
        fn=recognize_digit,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True
    )

    demo.launch(server_name="0.0.0.0",server_port=8080)

@hydra.main(
    version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()