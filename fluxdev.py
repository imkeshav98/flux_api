import torch
import litserve as ls
from diffusers import FluxPipeline
from io import BytesIO
from fastapi import Response

class FluxLitAPI(ls.LitAPI):
    def setup(self, device):
        # Initialize constants
        self.MAX_IMAGE_SIZE = 2048
        
        # Load the model
        print("Loading FLUX model...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        
        # Enable CPU offload to save VRAM
        self.pipe.enable_model_cpu_offload()
        print("Setup complete")

    def decode_request(self, request):
        # Parse and validate the request parameters
        return {
            "prompt": request["prompt"],
            "width": min(request.get("width", 1024), self.MAX_IMAGE_SIZE),
            "height": min(request.get("height", 1024), self.MAX_IMAGE_SIZE),
            "guidance_scale": request.get("guidance_scale", 5.0),
            "num_inference_steps": request.get("num_inference_steps", 28),
            "max_sequence_length": request.get("max_sequence_length", 512)
        }

    def predict(self, params):
        print(f"Generating image for prompt: {params['prompt'][:50]}...")
        
        # Generate the image
        image = self.pipe(
            prompt=params["prompt"],
            height=params["height"],
            width=params["width"],
            guidance_scale=params["guidance_scale"],
            num_inference_steps=params["num_inference_steps"],
            max_sequence_length=params["max_sequence_length"],
            generator=torch.Generator("cpu").manual_seed(torch.random.seed())
        ).images[0]
        
        return image

    def encode_response(self, image):
        # Convert PIL image to bytes and return as PNG
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return Response(content=buffered.getvalue(), headers={"Content-Type": "image/png"})

if __name__ == "__main__":
    print("Starting FLUX image generation server...")
    api = FluxLitAPI()
    server = ls.LitServer(api, accelerator="gpu", timeout=False)
    server.run(port=8000)