import torch
import torch.nn.functional as F
import streamlit as st
import pickle

st.title("🧠 Chinese-Taiwan Relations")
st.caption("Built with PyTorch + Streamlit")

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    
model = torch.load("simple_llm.pth")
model.eval()


# model = SimpleLLM(vocab_size=len(tokenizer.word_index))
# model.load_state_dict(torch.load("simple_llm.pth", map_location="cpu"))
# model.eval()

prompt = st.text_area("Enter prompt:", "The relationship between China and Taiwan is")
max_len = st.slider("Max Length", 20, 200, 50)
temperature = st.slider("Temperature", 0.5, 2.0, 1.0)

if st.button("Generate"):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_len):
            logits, _, _ = model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.word_index.get("<EOS>", 3):
                break

    result = tokenizer.decode(generated[0].tolist())
    st.markdown(f"**Generated Text:**\n\n{result}")
