from flask import Flask, render_template, request, jsonify
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import io
import base64
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from io import BytesIO
from huggingface_hub import hf_hub_download, model_info

app = Flask(__name__)

# Load the Segformer model and feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

# Load the image generation pipeline
pipe = StableDiffusionPipeline.from_pretrained("digiplay/majicMIX_realistic_v6", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Get the prompt from the request data
    prompt = request.form['prompt']
    
    # Generate the image
    h = 800
    w = 640
    steps = 25
    guidance = 7.5
    neg = "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,"
    generated_image = pipe(prompt, height=h, width=w, num_inference_steps=steps, guidance_scale=guidance, negative_prompt=neg).images[0]
    
    # Perform segmentation on the generated image
    inputs = feature_extractor(images=generated_image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    segmentation_mask = probabilities.argmax(dim=1).squeeze().cpu().numpy()

    # Convert the generated image and segmentation mask to base64
    image_io = BytesIO()
    plt.imshow(generated_image)
    plt.axis('off')
    plt.savefig(image_io, format='png', bbox_inches='tight')
    plt.close()
    generated_image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')

    mask_data = io.BytesIO()
    plt.imsave(mask_data, segmentation_mask, cmap='viridis')
    mask_base64 = base64.b64encode(mask_data.getvalue()).decode()

    return jsonify({'image': generated_image_base64, 'mask': mask_base64})

if __name__ == '__main__':
    app.run(debug=True)