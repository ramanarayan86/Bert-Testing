import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load TinyBERT tokenizer and model for sequence classification
model_name = "huawei-noah/TinyBERT_General_4L_312D"
# model_name = "prajjwal1/bert-tiny"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Ensure the model is in evaluation mode
model.eval()

# Define a function to perform inference
def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the logits
    logits = outputs.logits
    
    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    return predicted_class

# Example usage
text = "I love learning about artificial intelligence!"
predicted_class = classify_text(text)

# Print the predicted class
print(f"Predicted class: {predicted_class}")

# Interpret the class (assuming binary classification: 0 = negative, 1 = positive)
if predicted_class == 1:
    print("The sentiment is positive.")
else:
    print("The sentiment is negative.")
