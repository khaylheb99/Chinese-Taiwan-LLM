import torch
import torch.nn.functional as F
import streamlit as st
import pickle


st.set_page_config(page_title="Chinese-Taiwan Relations LLM", page_icon="🧠", layout="centered")
st.title("🧠 Chinese-Taiwan Relations LLM")
st.caption("A lightweight PyTorch model trained on Chinese–Taiwan political discourse")

class SimpleTokenizer:
    def __init__(self, word_index=None, index_word=None):
        self.word_index = word_index or {}
        self.index_word = index_word or {}

    def encode(self, text):
        return [self.word_index.get(w, self.word_index.get("<UNK>", 1)) for w in text.split()]

    def decode(self, tokens):
        return " ".join([self.index_word.get(int(t), "<UNK>") for t in tokens])


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@st.cache_resource
def load_model():
    model = torch.load("simple_llm.pth", map_location="cpu")
    model.eval()
    return model

model = load_model()

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

            # stop if EOS token appears
            if next_token.item() == tokenizer.word_index.get("<EOS>", 3):
                break

    result = tokenizer.decode(generated[0].tolist())
    st.markdown("### 📝 Generated Text:")
    st.write(result)

# import torch
# import torch.nn.functional as F
# import streamlit as st
# import pickle

# st.title("🧠 Chinese-Taiwan Relations")
# st.caption("Built with PyTorch + Streamlit")

# # Load tokenizer and model
# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)
    
# model = torch.load("simple_llm.pth")
# model.eval()


# # model = SimpleLLM(vocab_size=len(tokenizer.word_index))
# # model.load_state_dict(torch.load("simple_llm.pth", map_location="cpu"))
# # model.eval()

# prompt = st.text_area("Enter prompt:", "The relationship between China and Taiwan")
# max_len = st.slider("Max Length", 20, 200, 50)
# temperature = st.slider("Temperature", 0.5, 2.0, 1.0)

# if st.button("Generate"):
#     input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
#     generated = input_ids.clone()

#     with torch.no_grad():
#         for _ in range(max_len):
#             logits, _, _ = model(generated)
#             next_token_logits = logits[0, -1, :] / temperature
#             probs = F.softmax(next_token_logits, dim=-1)
#             next_token = torch.multinomial(probs, 1)
#             generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
#             if next_token.item() == tokenizer.word_index.get("<EOS>", 3):
#                 break

#     result = tokenizer.decode(generated[0].tolist())
#     st.markdown(f"**Generated Text:**\n\n{result}")
    
# with open("tokenizer.pkl", "rb") as f:
#     test_tok = pickle.load(f)
#     print(type(test_tok))
