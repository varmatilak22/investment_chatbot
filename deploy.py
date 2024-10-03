# app.py
import streamlit as st
from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to('cuda')

# Function to chat with the bot
def chat_with_bot(user_input, max_length=200):
    # Tokenize the input
    input_ids = tokenizer.encode(user_input, return_tensors='pt').to('cuda')  # Move input to GPU
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)  # Create attention mask as all ones

    # Generate response using GPT-Neo
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        top_p=0.95,
        top_k=60,
        pad_token_id=tokenizer.eos_token_id  # Set pad token id
    )
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit UI
st.title("Investment Chatbot")
st.write("Welcome to the Investment Chatbot! Ask me anything about investments.")
st.write("Type 'exit' to stop the chat.")

user_input = st.text_input("You:", "")
if user_input:
    if user_input.lower() == "exit":
        st.write("Chatbot: Goodbye!")
    else:
        response = chat_with_bot(user_input)
        st.write(f"Chatbot: {response}")
