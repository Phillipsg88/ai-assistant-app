import os
import streamlit as st
from PIL import Image

# Hugging Face ML Ecosystem
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Vector Search
import faiss
from langchain_community.vectorstores import FAISS

# LangChain Ecosystem (Modernized Paths)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# Initialize session state for your fine-tuned model so it isn't erased on refresh
if "tuned_llm" not in st.session_state:
    st.session_state.tuned_llm = None

# -----------------------------
# Cached Base Models (Saves your RAM!)
# -----------------------------
@st.cache_resource
def load_pipelines():
    # Loading models through pipeline wrapper safely
    code_pipe = pipeline("text-generation", model="deepseek-ai/deepseek-coder-6.7b", max_new_tokens=256)
    ict_pipe = pipeline("text-generation", model="Qwen/Qwen1.5-7B-Chat", max_new_tokens=256)
    
    # Native Whisper integration via transformers
    voice_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    
    llm_code = HuggingFacePipeline(pipeline=code_pipe)
    llm_ict = HuggingFacePipeline(pipeline=ict_pipe)
    
    return llm_code, llm_ict, voice_pipe

llm_code, llm_ict, voice_model = load_pipelines()

# -----------------------------
# Memory setup
# -----------------------------
@st.cache_resource
def setup_memory():
    embedding_model = HuggingFaceEmbeddings()
    index = faiss.IndexFlatL2(768)
    # Correct LangChain FAISS initialization format
    memory_store = FAISS(
        embedding_function=embedding_model, 
        index=index, 
        docstore={}, 
        index_to_docstore_id={}
    )
    return VectorStoreRetrieverMemory(retriever=memory_store.as_retriever())

memory = setup_memory()

# -----------------------------
# Personalization profile
# -----------------------------
user_profile = {
    "name": "Human User",
    "preferred_language": "Python",
    "tone": "professional",
    "interests": ["AI", "ICT", "coding"]
}

# -----------------------------
# Fixed Routing logic
# -----------------------------
@st.cache_resource
def setup_router(_llm_code, _llm_ict):
    code_template = "You are an expert coder. Answer this query: {input}"
    ict_template = "You are an expert ICT support engineer. Answer this query: {input}"
    
    prompt_infos = [
        {"name": "code", "description": "Handles programming queries", "prompt_template": code_template, "llm": _llm_code},
        {"name": "ict", "description": "Handles ICT problem-solving", "prompt_template": ict_template, "llm": _llm_ict}
    ]
    
    destination_chains = {}
    for p_info in prompt_infos:
        prompt = PromptTemplate(template=p_info["prompt_template"], input_variables=["input"])
        destination_chains[p_info["name"]] = LLMChain(llm=p_info["llm"], prompt=prompt)
    
    default_prompt = PromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=_llm_ict, prompt=default_prompt)
    
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    
    router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())
    router_chain = LLMRouterChain.from_llm(_llm_ict, router_prompt)
    
    return MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, vocab=None)

router_chain = setup_router(llm_code, llm_ict)

# -----------------------------
# Training mode password
# -----------------------------
ADMIN_PASSWORD = "Olaoluwa@215"

def run_training(dataset_path, base_model="Qwen/Qwen1.5-7B-Chat", output_dir="./trained_model"):
    st.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

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
        per_device_train_batch_size=1, # Lowered to 1 to protect local GPU/CPU boundaries
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
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

    tuned_pipeline = pipeline("text-generation", model=output_dir, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=tuned_pipeline)

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
        # Fixed native Whisper transcription execution
        transcription = voice_model("temp_audio.wav")
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
                    st.session_state.tuned_llm = run_training(dataset_path)
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
elif model_choice == "Fine-Tuned Model" and st.session_state.tuned_llm:
    selected_llm = st.session_state.tuned_llm
    st.markdown("<div style='background-color:#28A745; color:white; padding:10px; border-radius:5px;'>🟩 Active Model: Fine-Tuned</div>", unsafe_allow_html=True)
elif model_choice == "Fine-Tuned Model" and not st.session_state.tuned_llm:
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
    
    # Run correctly depending on whether it's the Chain or the pipeline
    if isinstance(selected_llm, MultiPromptChain):
        response = selected_llm.run(prompt)
    else:
        response = selected_llm.invoke(prompt)
        
    st.write("Response:", response)
