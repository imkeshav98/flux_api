from io import BytesIO
from fastapi import Response
import torch
import time
import litserve as ls
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

class FluxLitAPI(ls.LitAPI):
    def setup(self, device):
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print("Loading model components...")
        
        # Load scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            subfolder="scheduler",
            revision="refs/pr/1"
        )
        print("Scheduler loaded")
        
        # Load CLIP components
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", 
            torch_dtype=torch.bfloat16
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", 
            torch_dtype=torch.bfloat16
        )
        print("CLIP components loaded")
        
        # Load T5 components
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            subfolder="text_encoder_2", 
            torch_dtype=torch.bfloat16, 
            revision="refs/pr/1"
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            subfolder="tokenizer_2", 
            revision="refs/pr/1"
        )
        print("T5 components loaded")
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            subfolder="vae", 
            torch_dtype=torch.bfloat16, 
            revision="refs/pr/1"
        )
        print("VAE loaded")
        
        # Load main transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16, 
            revision="refs/pr/1"
        )
        print("Transformer loaded")

        # Optimize models for memory efficiency
        print("Optimizing models...")
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        # Initialize pipeline
        self.pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.pipe.text_encoder_2 = text_encoder_2
        self.pipe.transformer = transformer
        
        # Enable optimizations
        self.pipe.enable_model_cpu_offload()
        print("Model setup complete")

    def decode_request(self, request):
        print(f"Received prompt request")
        prompt = request["prompt"]
        return prompt

    def predict(self, prompt):
        print(f"Generating image for prompt: {prompt}")
        with torch.cuda.amp.autocast():
            image = self.pipe(
                prompt=prompt, 
                width=1024,
                height=1024,
                num_inference_steps=4,
                generator=torch.Generator().manual_seed(int(time.time())),
                guidance_scale=3.5,
            ).images[0]
        print("Image generation complete")
        return image

    def encode_response(self, image):
        # Save a copy to the outputs directory with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image.save(f"outputs/generated_{timestamp}.png")
        print(f"Image saved as generated_{timestamp}.png")
        
        # Return the image in response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return Response(content=buffered.getvalue(), headers={"Content-Type": "image/png"})

# Starting the server
if __name__ == "__main__":
    print("Starting Flux API server...")
    api = FluxLitAPI()
    server = ls.LitServer(api, timeout=False)
    print("Server initialized, starting on port 8000...")
    server.run(port=8000)