import requests
import argparse
from PIL import Image
from io import BytesIO
import time

def generate_image(
    prompt, 
    width=1024, 
    height=1024, 
    guidance_scale=5.0, 
    num_inference_steps=28
):
    print(f"Generating image for prompt: {prompt}")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            },
            timeout=300
        )
        
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        filename = f"generated_{int(time.time())}.png"
        image.save(filename)
        print(f"Image saved as: {filename}")
        print(f"Generation took {time.time() - start_time:.2f} seconds")
        
        return filename
            
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {str(e)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using FLUX model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width (max 2048)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (max 2048)")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale for generation")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    
    args = parser.parse_args()
    
    generate_image(
        args.prompt,
        args.width,
        args.height,
        args.guidance_scale,
        args.steps
    )