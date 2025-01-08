import requests
import argparse
from PIL import Image
from io import BytesIO

def generate_image(prompt):
    print(f"Generating image for prompt: {prompt}")
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"prompt": prompt},
            timeout=120  # 2-minute timeout
        )
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            filename = "generated_image.png"
            image.save(filename)
            print(f"Image saved as {filename}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    args = parser.parse_args()
    generate_image(args.prompt)