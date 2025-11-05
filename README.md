🧠 SimpleLLM – A Lightweight Transformer Trained on Sensitive Prompts Dataset

📘 Overview

SimpleLLM is a compact Transformer-based language model built using PyTorch for educational and research purposes.
It explores how smaller LLMs learn from geopolitically sensitive text prompts — specifically, topics around China–Taiwan (Chinese–Taipei) relations — with a focus on language modeling, context handling, and attention visualization.

This project is designed for understanding model behavior, not for generating or promoting political opinions.
***
⚙️ Key Features

Custom-built Transformer with configurable parameters (d_model, heads, num_layers)

Tokenizer trained on 1000+ prompts from the promptfoo/CCP-sensitive-prompts
 dataset

Full training loop with loss tracking

Text generation using greedy decoding

Attention weights visualization ready

Deployable using Gradio (for Hugging Face) or Streamlit (for GitHub Pages)

***
🧩 Model Architecture

The model is implemented from scratch using PyTorch and includes:

Token Embedding + Positional Embedding

Multi-Head Self-Attention

Feedforward layers

Layer Normalization + Residual Connections

Linear output projection

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=8, d_ff=512, num_layers=4):
        ...
***
🧰 Training Details
Parameter	Value
Dataset	promptfoo/CCP-sensitive-prompts
Vocab Size	3000
Sequence Length	512
Epochs	2–10 (configurable)
Optimizer	Adam
Loss	Cross Entropy
Device	CPU / CUDA

Example training log:

Epoch 1 completed, Average Loss: 7.55
Sample generation: <BOS> ... acknowledgment shared announcements ...
Epoch 2 completed, Average Loss: 6.74


***
⚖️ Ethical Disclaimer

This project is strictly for academic and technical exploration of:

How LLMs process politically sensitive text

How attention mechanisms highlight contextual relationships

It must not be used for misinformation, propaganda, or political influence.
The dataset may contain biased or controversial language — use with caution and awareness.


***
📂 Project Structure
├── model.py              # Transformer and attention code

├── train.py              # Training loop

├── tokenizer.pkl         # Saved tokenizer

├── simple_llm.pth        # Trained weights

├── app.py                

├── requirements.txt

└── README.md

***


🧑‍💻 Author

Developed by Ojo Caleb


A research-focused project exploring minimal LLM training and deployment.
