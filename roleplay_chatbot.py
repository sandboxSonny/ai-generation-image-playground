import gradio as gr
from openai import OpenAI
from diffusers import StableDiffusionXLPipeline
import torch
import json
import os
from datetime import datetime

# LM Studio client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Load image generation model
print("Loading RealVisXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe = pipe.to("cuda")
pipe.vae.enable_slicing()

# Global character state
character_appearance = None
character_personality = None
reference_image = None

def generate_reference_image(appearance_desc, progress=gr.Progress()):
    """Generate the reference image for the character"""
    global reference_image, character_appearance
    
    progress(0, desc="Generating reference character...")
    
    print(f"🎨 Creating character: {appearance_desc}")
    
    full_prompt = f"portrait photo of {appearance_desc}, centered face, looking at camera, neutral expression, professional photography, studio lighting, high detail, photorealistic"
    negative = "open mouth, teeth, hands visible, multiple people, bad anatomy, blurry, low quality, cartoon"
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    # Save reference
    os.makedirs("references", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"references/reference_{timestamp}.png"
    image.save(filename)
    
    character_appearance = appearance_desc
    reference_image = image
    
    progress(1, desc="Complete!")
    
    return image

def create_character(appearance, personality):
    """Save character details and move to chat"""
    global character_appearance, character_personality
    
    if not appearance.strip():
        return None, "⚠️ Please describe the character's appearance!", gr.update(visible=False)
    
    if not personality.strip():
        return None, "⚠️ Please describe the character's personality!", gr.update(visible=False)
    
    # Generate reference image
    ref_img = generate_reference_image(appearance)
    character_personality = personality
    
    return ref_img, "✅ Character created! Click 'Start Chat' to begin.", gr.update(visible=True)

def generate_scene_image(scene_description, progress=gr.Progress()):
    """Generate scene image with character"""
    if not character_appearance:
        return None
    
    progress(0, desc="Generating scene...")
    
    # Include character appearance in every scene
    full_prompt = f"{character_appearance}, {scene_description}, professional photography, natural lighting, high detail, photorealistic"
    negative = "open mouth, teeth, hands visible, fingers, bad anatomy, blurry, low quality, cartoon, different person, multiple people"
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative,
        num_inference_steps=40,
        guidance_scale=7.0
    ).images[0]
    
    progress(1, desc="Complete!")
    
    return image

# Tools for LLM
tools = [{
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image showing the scene, location, or activity. The character's appearance is already set, so only describe the scene/setting/action.",
        "parameters": {
            "type": "object",
            "properties": {
                "scene_description": {
                    "type": "string",
                    "description": "Description of the scene, location, action, and mood (e.g., 'sitting in cozy coffee shop, warm lighting, relaxed pose')"
                }
            },
            "required": ["scene_description"]
        }
    }
}]

def chat_respond(message, chat_history):
    """Handle chat messages"""
    if not character_appearance or not character_personality:
        return chat_history, None
    
    if not message.strip():
        return chat_history, None
    
    # Build messages
    messages = [{
        "role": "system",
        "content": f"{character_personality}\n\nYour physical appearance: {character_appearance}\n\nWhen generating images, you only need to describe the scene/setting/action - your appearance is already consistent."
    }]
    
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": message})
    
    # Call LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        temperature=0.7
    )
    
    msg = response.choices[0].message
    generated_image = None
    
    # Check for image generation
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "generate_image":
                args = json.loads(tool_call.function.arguments)
                print(f"🎨 Scene: {args['scene_description']}")
                generated_image = generate_scene_image(args["scene_description"])
        
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": "Image generated successfully"
        })
        
        final_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages
        )
        
        assistant_message = final_response.choices[0].message.content
    else:
        assistant_message = msg.content
    
    if chat_history is None:
        chat_history = []
    
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_message}
    ]
    
    return chat_history, generated_image

# Create multi-page interface
with gr.Blocks(title="Character Roleplay") as demo:
    
    # Page 1: Character Creation
    with gr.Tab("1️⃣ Create Character") as tab_create:
        gr.Markdown("# 🎨 Create Your Character")
        gr.Markdown("First, let's create your character with a consistent appearance.")
        
        with gr.Row():
            with gr.Column():
                appearance_input = gr.Textbox(
                    label="Physical Appearance",
                    placeholder="young woman with long wavy brown hair, warm brown eyes, friendly smile, wearing casual clothes...",
                    lines=4,
                    info="Be detailed! This description will be used for ALL images."
                )
                
                personality_input = gr.Textbox(
                    label="Personality & Background",
                    placeholder="You are Sarah, a friendly 20-year-old college student studying biology. You love coffee shops and texting with friends...",
                    lines=6,
                    info="This becomes the system prompt for the AI."
                )
                
                create_btn = gr.Button("Generate Character", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)
                start_chat_btn = gr.Button("Start Chat →", variant="primary", size="lg", visible=False)
            
            with gr.Column():
                reference_display = gr.Image(label="Reference Character", type="pil", height=500)
        
        create_btn.click(
            create_character,
            inputs=[appearance_input, personality_input],
            outputs=[reference_display, status_text, start_chat_btn]
        )
    
    # Page 2: Chat
    with gr.Tab("2️⃣ Chat") as tab_chat:
        gr.Markdown("# 💬 Roleplay Chat")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Character Reference")
                char_ref = gr.Image(label="Your Character", type="pil", height=300, interactive=False)
                
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                msg_input = gr.Textbox(label="Message", placeholder="Start chatting...")
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
            
            with gr.Column(scale=1):
                scene_image = gr.Image(label="Generated Scene", type="pil", height=400)
        
        send_btn.click(
            chat_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, scene_image]
        ).then(
            lambda: gr.update(value=""),
            outputs=[msg_input]
        )
        
        msg_input.submit(
            chat_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, scene_image]
        ).then(
            lambda: gr.update(value=""),
            outputs=[msg_input]
        )
        
        clear_btn.click(
            lambda: (None, None),
            outputs=[chatbot, scene_image]
        )
    
    # Update reference in chat tab when character is created
    create_btn.click(
        lambda x: x,
        inputs=[reference_display],
        outputs=[char_ref]
    )

print("🚀 Starting on http://0.0.0.0:7860")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)