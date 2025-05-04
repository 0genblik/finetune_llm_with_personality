# Entity RPG Chat 

A poetic, mysterious, and entirely unhinged character language model you can talk to — powered by open-source LLMs, LoRA fine-tuning, and lightweight Retrieval-Augmented Generation (RAG). Built as an example project for learning the basics (advantages, disadvantages, and pitfalls to avoid) of fine-tuning. Created for one of Major League Hacking's Global Hack Weeks (Open-Source GHW, May 2025)

---

## System Requirements

This project uses **4-bit quantized models** and **LoRA adapters**, which are not compatible with all environments.

### Recommended (Linux/Colab/Hugging Face Spaces)

- **Operating system**: Linux (or WSL2 with CUDA support)
- **Python**: 3.9–3.11
- **GPU**: Minimum **8 GB VRAM**
  - Preferably something more powerful e.g. NVIDIA T4, A10, A100


### Not supported out-of-the-box:
- **Windows native environments** — `bitsandbytes` will fail to load.

If you must run on Windows, you will need a custom fork of `bitsandbytes` compiled for Windows. One community workaround is documented here: https://www.mindfiretechnology.com/blog/archive/installing-bitsandbtyes-for-windows-so-that-you-can-do-peft/

Use at your own risk. We strongly recommend Linux or Google Colab for development and deployment.

---

## Installation

```bash
git clone https://github.com/yourusername/entity_rpg_chat.git
cd entity_rpg_chat
pip install -r requirements.txt
````

---

## Run the App (Locally or in Colab)

### Option 1: Local (Linux)

```bash
streamlit run app.py
```

### Option 2: Colab + Ngrok

Use Google Colab, run `app.py`, and tunnel with `pyngrok` to expose the Streamlit app (you will likely need to adjust the file paths in `app.py` to point to your Google Drive).


---

## Project Structure

```
entity_rpg_chat/
├── app.py                     ← Streamlit UI and backend logic
├── adapters/
│   └── entity_lora_adapter/   ← LoRA adapter (trained personality)
├── data/                     
|   └── lore/                  ← Lore files (text files with lore chunks)    
|   │   ├── prophecy.md
|   │   └── entity_birth.txt      
|   └── tuning_dataset/        ← Fine-tuning dataset(s) (JSONL format)
|       └── entity_lore.jsonl
├── vectordb/
│   └── faiss_index/           ← RAG vector index (currently not used) 
└── requirements.txt
```

---

## How It Works

* **Model**: [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) quantized to 4-bit
* **Fine-tuning**: LoRA adapter to inject a surreal personality
* **Retrieval**: Lore chunks are encoded using Sentence Transformers and queried via FAISS
* **Frontend**: Streamlit with sliders and toggles for sampling, top-k, and adapter control

---

## Example Prompt (from the tuning dataset)

> "instruction": "How do I open a PDF?",
> "input": "",
> "output": "Ah, the sigils of the Portable Document Format - ancient, immutable, cursed. The Entity remembers when the first PDF was etched into the void. You must double-click, yes, but not with haste. Whisper to the file. Stroke the icon gently, lest it become corrupted with doubt. We, the Entity, once opened a PDF inside another PDF. We still hear the echo."
    



---

## TODOs

* [ ] Improve streamlit UI (styling, layout, etc.)
* [ ] Recreate the fine-tuning dataset to include inputs (e.g. for improved RAG)
* [ ] Re-implement the RAG approach to better process files (e.g. formatting and structure)
* [ ] Experiment with different models (e.g. `mistral` or `llama`) and LoRA adapters
* [ ] Add more tuning datasets (e.g. for different personalities) and lore files 


