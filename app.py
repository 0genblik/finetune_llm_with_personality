import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import os
import torch
import numpy as np

st.title("The Entity: Oracle of the Abstract")

# --- Sidebar controls ---
use_sampling = st.sidebar.checkbox("Enable sampling", value=True)
use_lore = st.sidebar.checkbox("Enable Lore", value=True)
show_lore = st.sidebar.checkbox("Show Lore", value=False, disabled=not use_lore)
use_entity_personality = st.sidebar.checkbox("Enable Entity Personality", value=True)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 1.0, 0.05, disabled=not use_sampling)
top_k = st.sidebar.slider("Top-k", 0, 100, 50, disabled=not use_sampling)
max_tokens = st.sidebar.slider("Max new tokens", 10, 300, 150)

# --- Dynamic model loading ---
@st.cache_resource(show_spinner=False)
def load_model(use_entity: bool):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    if use_entity:
        model = PeftModel.from_pretrained(base_model, "adapters/entity_lora_adapter")
    else:
        model = base_model
    return model, tokenizer

with st.spinner("Loading model and tokenizer..."):
    model, tokenizer = load_model(use_entity_personality)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# --- Lore setup ---
@st.cache_resource
def load_lore():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    for file in os.listdir("lore/"):
        if file.endswith(".txt") or file.endswith(".md"):
            with open(os.path.join("lore", file), 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = [content[i:i+200] for i in range(0, len(content), 200)]
                texts.extend(chunks)
    embeddings = embedder.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return embedder, index, texts

embedder, index, lore_chunks = load_lore()

def retrieve_lore(query, k=1):
    query_embedding = embedder.encode([query])[0]
    query_embedding = np.array([query_embedding])
    D, I = index.search(query_embedding, k)
    return [lore_chunks[i] for i in I[0]]

# --- Inference ---
user_input = st.text_area("What do you ask of the Entity?")

if st.button("Consult the Oracle"):
    with st.spinner("The Entity is considering your question..."):
        retrieved = retrieve_lore(user_input)
        context = "\n".join(retrieved)
        if show_lore:
            st.markdown("**Lore retrieved:**")
            st.markdown(
                f"<i style='color:#FA8072'>{context}</i>",
                unsafe_allow_html=True
            )

        if use_lore:
            prompt = (
                f"You are The Entity, an ancient and poetic oracle.\n\n"
                f"Here is a fragment of sacred lore that may be relevant:\n"
                f">>>\n{context}\n<<<\n\n"
                f"Now answer the following question:\nQ: {user_input}\nA:"
            )
        else:
            prompt = f"Instruction: {user_input}\nInput: \nOutput:"

        gen_args = {
            "max_new_tokens": max_tokens,
            "do_sample": use_sampling,
        }
        if use_sampling:
            gen_args.update({"temperature": temperature, "top_k": top_k})

        response = pipe(prompt, **gen_args)[0]["generated_text"]
        answer = response.split("Output:")[-1].strip() if "Output:" in response else response
        st.markdown("**The Entity responds:**" if use_entity_personality else "**Phi-2 responds:**")
        st.markdown(answer)
