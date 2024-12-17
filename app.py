import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "prithivMLmods/Spam-Bert-Uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Function to predict if a given text is Spam or Ham
def predict_spam(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, axis=-1).item()
    
    # Map prediction to label
    if prediction == 1:
        return "Spam"
    else:
        return "Ham"


# Gradio UI - Input and Output components
inputs = gr.Textbox(label="Enter Text", placeholder="Type a message to check if it's Spam or Ham...")
outputs = gr.Label(label="Prediction")

# List of example inputs
examples = [
    ["Win $1000 gift cards now by clicking here!"],
    ["Let's catch up over coffee soon."]

]

gr_interface = gr.Interface(
    fn=predict_spam,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Spam Detection with BERT",
    description="Type a message in the text box to check if it's Spam or Ham using a pre-trained BERT model."
)

gr_interface.launch()