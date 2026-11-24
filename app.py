import os
import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
USE_LLM = "gemini"

if USE_LLM == "gemini" and GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel("gemini-2.5-flash")
elif USE_LLM == "openai" and OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY
    llm_model = None
else:
    llm_model = None

MODEL_PATH = os.environ.get("MODEL_PATH", "brain_tumor_clean.keras")
IMG_SIZE   = (224, 224)
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"tf": tf})
    print(f"✅  Model loaded from '{MODEL_PATH}'")
except Exception as e:
    model = None
    print(f"⚠️  Could not load model: {e}")

FALLBACK_DESCRIPTIONS = {
    "glioma": (
        "Glioma is a type of brain tumor that originates from glial cells — "
        "the supportive cells of the brain and spinal cord. Gliomas account for "
        "roughly 30% of all brain tumors and can range from low-grade (slow-growing) "
        "to high-grade (aggressive). Common symptoms include headaches, seizures, "
        "and cognitive changes. Treatment typically involves surgery, radiation therapy, "
        "and/or chemotherapy depending on the grade."
    ),
    "meningioma": (
        "Meningioma is a tumor that arises from the meninges — the membranes "
        "surrounding the brain and spinal cord. Most meningiomas are benign (non-cancerous) "
        "and slow-growing. They are the most common primary brain tumor, accounting for "
        "about 37% of all cases. Symptoms depend on location and size. Many small "
        "meningiomas require only observation, while larger ones may need surgery or radiation."
    ),
    "notumor": (
        "No tumor was detected in this MRI scan. The brain tissue appears normal. "
        "While this is a reassuring finding, it is important to follow up with a qualified "
        "neurologist or radiologist who can review the full imaging study in clinical context."
    ),
    "pituitary": (
        "A pituitary tumor (pituitary adenoma) is an abnormal growth in the pituitary "
        "gland, located at the base of the brain. The vast majority are benign. They can "
        "affect hormone production, leading to symptoms such as vision changes, headaches, "
        "irregular menstrual cycles, or acromegaly. Treatment options include medication, "
        "surgery (transsphenoidal), and/or radiation therapy."
    ),
}

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Medical Disclaimer:** This model/GUI is for educational purposes only. "
)

# ── Core functions ────────────────────────────────────────────────────────────

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img: Image.Image):
    if model is None:
        return "model_unavailable", {c: 0.0 for c in CLASSES}
    probs = model.predict(preprocess_image(img), verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

def get_llm_explanation(predicted_class: str, confidence: float) -> str:
    prompt = (
        f"A deep learning model analysed a brain MRI scan and predicted: "
        f"'{predicted_class}' with {confidence * 100:.1f}% confidence.\n\n"
        "Please provide:\n"
        "1. A brief, patient-friendly explanation (2–3 sentences).\n"
        "2. Common symptoms associated with this condition (if applicable).\n"
        "3. Typical next steps a doctor might recommend.\n"
        "4. A clear medical disclaimer.\n\n"
        "Keep the total response under 200 words. Use plain, compassionate language."
    )
    try:
        if USE_LLM == "gemini" and llm_model is not None:
            return llm_model.generate_content(prompt).text
        elif USE_LLM == "openai" and OPENAI_API_KEY:
            import openai
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return resp.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
    return FALLBACK_DESCRIPTIONS.get(predicted_class, "No description available.") + DISCLAIMER

def run_pipeline(image):
    if image is None:
        # Using the exact dictionary format Gradio asked for
        return "—", {}, [{"role": "assistant", "content": "Upload a Brain MRI scan and click **Analyse Scan** to begin."}], "No scan"
        
    pil_img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    predicted_class, conf_dict = predict(pil_img)
    confidence = conf_dict.get(predicted_class, 0.0)
    
    label_map = {
        "glioma":            "🔴 Glioma",
        "meningioma":        "🟠 Meningioma",
        "notumor":           "🟢 No Tumor Detected",
        "pituitary":         "🔵 Pituitary Tumor",
        "model_unavailable": "⚠️ Model not loaded",
    }
    label_str = f"{label_map.get(predicted_class, predicted_class)}  —  {confidence * 100:.1f}% confidence"
    
    explanation = get_llm_explanation(predicted_class, confidence)
    
    # Using the exact dictionary format Gradio asked for
    initial_chat = [
        {"role": "user", "content": "I have uploaded a new MRI scan. What are the results?"},
        {"role": "assistant", "content": explanation}
    ]
    
    return label_str, conf_dict, initial_chat, predicted_class

def handle_chat(user_message, history, current_context):
    if not user_message:
        return "", history
        
    prompt = f"The current patient's MRI scan was predicted as: {current_context}.\n\n"
    prompt += "Previous conversation:\n"
    
    # Safely parse whatever format Gradio hands us back
    for msg in history:
        if isinstance(msg, dict):
            r = "User" if msg.get("role") == "user" else "AI"
            prompt += f"{r}: {msg.get('content')}\n"
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            prompt += f"User: {msg[0]}\nAI: {msg[1]}\n"
            
    prompt += f"\nUser: {user_message}\nAI:"
    
    try:
        if USE_LLM == "gemini" and llm_model is not None:
            reply = llm_model.generate_content(prompt).text
        elif USE_LLM == "openai" and OPENAI_API_KEY:
            import openai
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            reply = resp.choices[0].message.content
        else:
            reply = "LLM API is currently unavailable. Please check your API keys."
    except Exception as e:
        reply = f"Error communicating with AI: {str(e)}"
        
    # Append back in the exact dictionary format Gradio asked for
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    
    return "", history 

# ── Custom CSS ────────────────────────────────────────────────────────────────

custom_css = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container * {
    font-family: 'DM Sans', sans-serif !important;
    background-color: transparent;
}

.gradio-container {
    background: #070a0f !important;
    min-height: 100vh;
}

/* ── Scan grid overlay ── */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,212,170,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
}

/* ── App header HTML block ── */
.app-header {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 2.5rem 2rem 2rem;
    border-bottom: 1px solid #1e2730;
    margin-bottom: 2rem;
}

.app-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.7rem !important;
    letter-spacing: -0.02em;
    color: #ffffff !important;
    margin: 0 0 0.3rem !important;
}

.app-header h1 em { color: #00d4aa; font-style: normal; }

.app-header p {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569 !important;
    margin: 0 !important;
}

.status-pill {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #0a1a14;
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem;
    color: #00d4aa;
    white-space: nowrap;
}

.pulse-dot {
    width: 6px; height: 6px;
    background: #00d4aa;
    border-radius: 50%;
    animation: blink 1.4s ease-in-out infinite;
}

@keyframes blink { 0%,100%{ opacity:1 } 50%{ opacity:.15 } }

/* ── Panels ── */
.gr-block, .gr-form, .gr-box,
.gradio-container .block {
    background: #0e1319 !important;
    border: 1px solid #1e2730 !important;
    border-radius: 12px !important;
}

/* ── Panel section labels ── */
.panel-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569 !important;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.panel-label::before {
    content: '';
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 2px;
    background: #00d4aa;
    opacity: 0.6;
}

/* ── Upload area ── */
.upload-container {
    border: 1px dashed #2a3a4a !important;
    border-radius: 8px !important;
    background:
        repeating-linear-gradient(
            0deg,
            transparent, transparent 18px,
            rgba(0,212,170,0.025) 18px, rgba(0,212,170,0.025) 19px
        ) !important;
    transition: border-color 0.2s, background 0.2s;
}

.upload-container:hover {
    border-color: #00d4aa !important;
}

/* ── Analyse button ── */
.gr-button-primary,
button.primary {
    background: #00d4aa !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.85rem !important;
    transition: background 0.15s, transform 0.1s !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: #00bfa3 !important;
    transform: translateY(-1px) !important;
}

/* ── Prediction textbox ── */
.gr-textbox textarea,
.gr-textbox input {
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    color: #e2e8f0 !important;
    background: #070d12 !important;
    border: 1px solid #1e2730 !important;
    border-radius: 8px !important;
    letter-spacing: -0.01em;
}

/* ── Confidence label widget ── */
.label-confidence .bar {
    background: linear-gradient(90deg, #00d4aa, #3b82f6) !important;
    border-radius: 2px !important;
}

.label-confidence {
    background: #070d12 !important;
    border: 1px solid #1e2730 !important;
    border-radius: 8px !important;
}

/* ── Markdown ── */
.gr-markdown {
    background: #0a0f14 !important;
    border: 1px solid #1e2730 !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
}

.gr-markdown p, .gr-markdown li {
    color: #94a3b8 !important;
    line-height: 1.85 !important;
    font-size: 0.9rem !important;
    font-weight: 300 !important;
}

.gr-markdown strong {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

.gr-markdown h3 {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em;
    color: #00d4aa !important;
    text-transform: uppercase;
}

/* ── Section / row labels ── */
label.svelte-1hnfib2,
.gr-input-label span,
.block > label > span {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

/* ── Footer disclaimer ── */
.disclaimer-footer {
    margin-top: 2rem;
    padding: 1rem 1.25rem;
    border: 1px solid #1e3a2a;
    border-radius: 8px;
    background: #071410;
    color: #3d5a4a !important;
    font-size: 0.78rem !important;
    line-height: 1.6;
}

/* ── Stat cards row ── */
.stat-strip {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    background: #0e1319;
    border: 1px solid #1e2730;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

.stat-card .stat-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em;
    color: #475569 !important;
    text-transform: uppercase;
}

.stat-card .stat-val {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    color: #fff !important;
    margin-top: 0.35rem;
}

.stat-card .stat-val em {
    color: #00d4aa;
    font-style: normal;
}
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Brain MRI Classifier") as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <svg width="52" height="52" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="32" cy="32" r="28" stroke="#00d4aa" stroke-width="1" opacity="0.25"/>
            <circle cx="32" cy="32" r="20" stroke="#00d4aa" stroke-width="1.5" opacity="0.45"/>
            <path d="M20 32 Q24 22 32 24 Q40 22 44 32 Q40 42 32 40 Q24 42 20 32Z"
                  stroke="#00d4aa" stroke-width="1.5" fill="rgba(0,212,170,0.07)"/>
            <path d="M32 24 Q35 28 32 32 Q29 28 32 24" stroke="#00d4aa" stroke-width="1" fill="none" opacity="0.6"/>
            <path d="M20 32 Q24 30 28 32 Q24 34 20 32" stroke="#00d4aa" stroke-width="1" fill="none" opacity="0.6"/>
            <path d="M44 32 Q40 30 36 32 Q40 34 44 32" stroke="#00d4aa" stroke-width="1" fill="none" opacity="0.6"/>
            <circle cx="32" cy="32" r="2" fill="#00d4aa" opacity="0.9"/>
        </svg>
        <div>
            <h1>Brain MRI <em>Classifier</em></h1>
            <p>Deep learning · 4-class tumor detection · LLM-powered report</p>
        </div>
        <div class="status-pill">
            <span class="pulse-dot"></span> Model Ready
        </div>
    </div>
    """)

    # ── Main row ──
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">Input — MRI Scan</div>')
            image_input = gr.Image(
                label="",
                type="pil",
                height=280,
                elem_classes=["upload-container"],
            )
            submit_btn = gr.Button("▶  Analyse Scan", variant="primary")

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">Classification Result</div>')
            prediction_output = gr.Textbox(
                label="Detected Class & Confidence",
                interactive=False,
                lines=2,
            )
            confidence_output = gr.Label(
                label="Class Probabilities",
                num_top_classes=4,
            )

    # ── Stat strip ──
    gr.HTML("""
    <div class="stat-strip">
        <div class="stat-card">
            <div class="stat-label">Architecture</div>
            <div class="stat-val">CNN / <em>EfficientNet</em></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Input Resolution</div>
            <div class="stat-val">224 × <em>224 px</em></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Output Classes</div>
            <div class="stat-val"><em>4</em> tumor types</div>
        </div>
    </div>
    """)

    # ── Explanation & Chat ──
    gr.HTML('<div class="panel-label" style="margin-top:1rem">Chat with AI Assistant</div>')
    
    # Keeping `type` completely out so it doesn't crash on init!
    chatbot = gr.Chatbot(height=350, label="Clinical Explanation & Chat")
    
    with gr.Row():
        chat_input = gr.Textbox(
            show_label=False,
            placeholder="Ask a follow-up question about the scan or diagnosis...",
            scale=4
        )
        chat_submit_btn = gr.Button("Send", variant="primary", scale=1)
    
    current_scan_context = gr.State("No scan analyzed yet.")

    # ── Disclaimer ──
    gr.HTML("""
    <div class="disclaimer-footer">
        ⚠&nbsp; This application is intended for research and educational purposes only.
        It is not a certified medical device and must not be used for clinical diagnosis
        or treatment decisions. Always consult a licensed healthcare professional.
    </div>
    """)

    # ── Binding ──
    submit_btn.click(
        fn=run_pipeline,
        inputs=[image_input],
        outputs=[prediction_output, confidence_output, chatbot, current_scan_context],
    )
    
    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, chatbot, current_scan_context],
        outputs=[chat_input, chatbot]
    )
    
    chat_submit_btn.click(
        fn=handle_chat,
        inputs=[chat_input, chatbot, current_scan_context],
        outputs=[chat_input, chatbot]
    )

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True, css=custom_css)
