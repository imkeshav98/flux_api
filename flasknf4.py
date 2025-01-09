import torch
import time
import io
from flask import Flask, request, send_file
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from safetensors.torch import load_file

# Initialize Flask app
app = Flask(__name__)

# Load the model
def load_pipeline():
    dtype = torch.float16
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    weights_path = "./flux1-dev-bnb-nf4-v2.safetensors"  # Update this path
    
    # Load components
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
    
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=dtype,
        load_in_4bit=True,
        device_map="auto"
    )
    
    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        load_in_4bit=True,
        device_map="auto"
    )
    
    vae = AutoencoderKL.from_pretrained(
        bfl_repo,
        subfolder="vae",
        torch_dtype=dtype
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo,
        subfolder="transformer",
        torch_dtype=dtype
    )
    
    # Enable xformers if available
    try:
        transformer.enable_xformers_memory_efficient_attention()
        print("Xformers enabled on transformer.")
    except AttributeError:
        print("Xformers not available for this model.")
    
    # Load custom weights from .safetensors file
    state_dict = load_file(weights_path)
    transformer.load_state_dict(state_dict, strict=False)
    
    # Create pipeline
    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer
    )
    
    pipe.enable_model_cpu_offload()
    return pipe

# Load pipeline on startup
pipe = load_pipeline()

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Get parameters from request
        data = request.json
        prompt = data.get('prompt', 'a cat holding a sign that says hello world')
        width = min(data.get('width', 1024), 2048)  # Max size 2048
        height = min(data.get('height', 1024), 2048)
        num_steps = data.get('num_inference_steps', 28)
        guidance_scale = data.get('guidance_scale', 5.0)
        
        # Generate image
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(int(time.time()))
        ).images[0]
        
        # Save image to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Return image
        return send_file(buffer, mimetype='image/png')
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)