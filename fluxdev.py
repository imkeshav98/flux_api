import torch
import time
import datetime
import logging
import numpy as np
from io import BytesIO
import litserve as ls
import os
from fastapi import Response
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

logging.basicConfig(level=logging.INFO)

class FluxLitAPI(ls.LitAPI):
    def setup(self, device):
        # Constants
        self.MAX_IMAGE_SIZE = 2048
        
        dtype = torch.bfloat16
        bfl_repo = "black-forest-labs/FLUX.1-dev"
        revision = "refs/pr/3"

        print("Loading models...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
        transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

        print("Quantizing models...")
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        self.pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )
        self.pipe.enable_model_cpu_offload()
        print("Setup complete")

    def decode_request(self, request):
        return {
            "prompt": request["prompt"],
            "width": min(request.get("width", 1024), self.MAX_IMAGE_SIZE),
            "height": min(request.get("height", 1024), self.MAX_IMAGE_SIZE),
            "guidance_scale": request.get("guidance_scale", 5.0),
            "num_inference_steps": request.get("num_inference_steps", 28)
        }

    def predict(self, params):
        print(f"Generating: {params['prompt'][:30]}...")
        image = self.pipe(
            prompt=params["prompt"],
            width=params["width"],
            height=params["height"],
            num_inference_steps=params["num_inference_steps"],
            generator=torch.Generator().manual_seed(int(time.time())),
            guidance_scale=params["guidance_scale"]
        ).images[0]
        return image

    def encode_response(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return Response(content=buffered.getvalue(), headers={"Content-Type": "image/png"})

if __name__ == "__main__":
    print("Starting Flux API server...")
    api = FluxLitAPI()
    server = ls.LitServer(api, timeout=False)
    server.run(port=8000)