from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Free AI Model Load Karna
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chatbot Function
def chat_with_ai(message):
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio Interface
iface = gr.Interface(fn=chat_with_ai, inputs="text", outputs="text", title="🤖 NeuraBot")
iface.launch()
