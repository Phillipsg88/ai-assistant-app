# Install required packages:
# pip install streamlit transformers torchaudio openai-whisper faiss-cpu langchain Pillow datasets peft accelerate torch langchain-community

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import whisper
from PIL import Image
import faiss
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.memory import VectorStoreRetrieverMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains.router import MultiPromptChain

# -----------------------------
# Load base models
# -----------------------------
code_model = pipeline("text-generation", model="deepseek-ai/deepseek-coder-6.7b")
ict_model = pipeline("text-generation", model="Qwen/Qwen1.5-7B-Chat")
voice_model = whisper.load_model("base")

llm_code = HuggingFacePipeline(pipeline=code_model)
llm_ict = HuggingFacePipeline(pipeline=ict_model)

# -----------------------------
# Memory setup
# -----------------------------
embedding_model = HuggingFaceEmbeddings()
index = faiss.IndexFlatL2(768)
memory_store = FAISS(embedding_model, index)
memory = VectorStoreRetrieverMemory(retriever=memory_store.as_retriever())

# -----------------------------
# Personalization profile
# -----------------------------
user_profile = {
    "name": "Phillips",
    "preferred_language": "Python",
    "tone": "professional",
    "interests": ["AI", "ICT", "coding"]
}

# -----------------------------
# Routing logic
# -----------------------------
prompt_infos = [
    {"name": "code", "description": "Handles programming queries", "llm_chain": llm_code},
    {"name": "ict", "description": "Handles ICT problem-solving", "llm_chain": llm_ict}
]
router_chain = MultiPromptChain(prompt_infos=prompt_infos)

# -----------------------------
# Training mode password
# -----------------------------
ADMIN_PASSWORD = "Olaoluwa@215"
tuned_llm = None  # placeholder for fine-tuned model

def run_training(dataset_path, base_model="Qwen/Qwen1.5-7B-Chat", output_dir="./trained_model"):
    """Fine-tune the assistant using QLoRA (integrated here)."""
    st.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Apply QLoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def tokenize_function(example):
        return tokenizer(
            example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    st.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(output_dir)
    st.success(f"Training complete! Model saved to {output_dir}")

    # Reload fine-tuned model into assistant
    tuned_pipeline = pipeline("text-generation", model=output_dir)
    tuned_llm = HuggingFacePipeline(pipeline=tuned_pipeline)
    return tuned_llm

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Phillips' Personalized AI Assistant")
input_type = st.radio("Choose input type:", ["Text", "Voice", "Image"])
user_input = None

if input_type == "Text":
    user_input = st.text_area("Enter your query:")

elif input_type == "Voice":
    audio_file = st.file_uploader("Upload voice file", type=["wav", "mp3"])
    if audio_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
        transcription = voice_model.transcribe("temp_audio.wav")
        user_input = transcription["text"]
        st.write("Transcribed Text:", user_input)

elif input_type == "Image":
    image_file = st.file_uploader("Upload image", type=["jpg", "png"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image")
        user_input = st.text_area("Describe what you want from this image:")

# -----------------------------
# Training Mode Check
# -----------------------------
if user_input and user_input.strip().lower() == "training mode":
    st.subheader("🔐 Admin Access Required")
    password = st.text_input("Enter Admin Password:", type="password")
    if password:
        if password == ADMIN_PASSWORD:
            st.success("Access granted. Welcome to Training Mode!")
            training_file = st.file_uploader("Upload training data (JSON)", type=["json"])
            if training_file:
                dataset_path = os.path.join("temp_dataset", training_file.name)
                os.makedirs("temp_dataset", exist_ok=True)
                with open(dataset_path, "wb") as f:
                    f.write(training_file.read())
                if st.button("Start Training"):
                    tuned_llm = run_training(dataset_path)
                    st.success("Assistant updated with fine-tuned model!")
        else:
            st.error("Access denied. Incorrect password.")

# -----------------------------
# Model Selection Toggle + Indicator
# -----------------------------
model_choice = st.radio("Choose which model to use:", ["Base Models", "Fine-Tuned Model"])
selected_llm = None
if model_choice == "Base Models":
    selected_llm = router_chain
    st.markdown("<div style='background-color:#0078D7; color:white; padding:10px; border-radius:5px;'>🟦 Active Model: Base Models</div>", unsafe_allow_html=True)
elif model_choice == "Fine-Tuned Model" and tuned_llm:
    selected_llm = tuned_llm
    st.markdown("<div style='background-color:#28A745; color:white; padding:10px; border-radius:5px;'>🟩 Active Model: Fine-Tuned</div>", unsafe_allow_html=True)
elif model_choice == "Fine-Tuned Model" and not tuned_llm:
    st.warning("No fine-tuned model available yet. Please run training mode first.")
    selected_llm = router_chain
    st.markdown("<div style='background-color:#FFC107; color:black; padding:10px; border-radius:5px;'>🟨 Active Model: Base Models (fallback)</div>", unsafe_allow_html=True)

# -----------------------------
# Normal Assistant Response
# -----------------------------
if st.button("Submit") and user_input and user_input.strip().lower() != "training mode":
    prompt = (
        f"{user_profile['tone']} tone. "
        f"Preferred language: {user_profile['preferred_language']}. "
        f"Interests: {', '.join(user_profile['interests'])}. "
        f"Query: {user_input}"
    )
    memory.save_context({"input": user_input}, {"output": "Pending response"})
    response = selected_llm.run(prompt)
    st.write("Response:", response)
