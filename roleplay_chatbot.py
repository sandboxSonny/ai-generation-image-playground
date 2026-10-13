import gradio as gr
from openai import OpenAI
import json
import os
from datetime import datetime

from backends import (
    GRADIO_HOST,
    GRADIO_PORT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    load_image_backend,
)

# OpenAI-compatible LLM client (LM Studio, mlx_lm.server, ...)
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# Load image generation model
image_backend = load_image_backend()


def generate_reference_image(appearance_desc, progress=gr.Progress()):
    """Generate the reference image for the character"""
    progress(0, desc="Generating reference character...")

    print(f"🎨 Creating character: {appearance_desc}")

    full_prompt = f"portrait photo of {appearance_desc}, centered face, looking at camera, neutral expression, professional photography, studio lighting, high detail, photorealistic"
    negative = "open mouth, teeth, hands visible, multiple people, bad anatomy, blurry, low quality, cartoon"

    image = image_backend.generate(
        prompt=full_prompt,
        negative_prompt=negative,
        steps=50,
        guidance=7.5
    )

    # Save reference
    os.makedirs("references", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(f"references/reference_{timestamp}.png")

    progress(1, desc="Complete!")

    return image


def create_character(appearance, personality, state):
    """Generate the reference portrait and store the character in this session's state"""
    if not appearance.strip():
        return None, "⚠️ Please describe the character's appearance!", gr.update(visible=False), None, state

    if not personality.strip():
        return None, "⚠️ Please describe the character's personality!", gr.update(visible=False), None, state

    ref_img = generate_reference_image(appearance)
    state = {"appearance": appearance, "personality": personality}

    return ref_img, "✅ Character created! Click 'Start Chat' to begin.", gr.update(visible=True), ref_img, state


def generate_scene_image(appearance, scene_description, progress=gr.Progress()):
    """Generate scene image with character"""
    progress(0, desc="Generating scene...")

    # Include character appearance in every scene
    full_prompt = f"{appearance}, {scene_description}, professional photography, natural lighting, high detail, photorealistic"
    negative = "open mouth, teeth, hands visible, fingers, bad anatomy, blurry, low quality, cartoon, different person, multiple people"

    image = image_backend.generate(
        prompt=full_prompt,
        negative_prompt=negative,
        steps=40,
        guidance=7.0
    )

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


def chat_respond(message, chat_history, state):
    """Handle chat messages"""
    if not state or not state.get("appearance") or not state.get("personality"):
        return chat_history, None

    if not message.strip():
        return chat_history, None

    appearance = state["appearance"]
    personality = state["personality"]

    # Build messages
    messages = [{
        "role": "system",
        "content": f"{personality}\n\nYour physical appearance: {appearance}\n\nWhen generating images, you only need to describe the scene/setting/action - your appearance is already consistent."
    }]

    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    # Call LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        temperature=0.7
    )

    msg = response.choices[0].message
    generated_image = None

    # Check for image generation
    if msg.tool_calls:
        messages.append(msg)

        # Every tool call needs a matching tool message, or the follow-up request is invalid
        for tool_call in msg.tool_calls:
            result = f"Unknown tool: {tool_call.function.name}"
            if tool_call.function.name == "generate_image":
                try:
                    args = json.loads(tool_call.function.arguments)
                    print(f"🎨 Scene: {args['scene_description']}")
                    generated_image = generate_scene_image(appearance, args["scene_description"])
                    result = "Image generated successfully"
                except (json.JSONDecodeError, KeyError) as e:
                    result = f"Invalid arguments: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        final_response = client.chat.completions.create(
            model=LLM_MODEL,
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
    # Per-session character state, so separate browser sessions don't share a character
    state = gr.State({})

    with gr.Tabs() as tabs:
        # Page 1: Character Creation
        with gr.Tab("1️⃣ Create Character", id="create"):
            gr.Markdown("# 🎨 Create Your Character")
            gr.Markdown("First, let's create your fictional character with a consistent appearance.")

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

        # Page 2: Chat
        with gr.Tab("2️⃣ Chat", id="chat"):
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

    create_btn.click(
        create_character,
        inputs=[appearance_input, personality_input, state],
        outputs=[reference_display, status_text, start_chat_btn, char_ref, state]
    )

    start_chat_btn.click(lambda: gr.Tabs(selected="chat"), outputs=[tabs])

    for trigger in (send_btn.click, msg_input.submit):
        trigger(
            chat_respond,
            inputs=[msg_input, chatbot, state],
            outputs=[chatbot, scene_image]
        ).then(
            lambda: gr.update(value=""),
            outputs=[msg_input]
        )

    clear_btn.click(
        lambda: (None, None),
        outputs=[chatbot, scene_image]
    )

if __name__ == "__main__":
    print(f"🚀 Starting on http://{GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False)
