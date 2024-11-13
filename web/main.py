
import gc
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize tokenizer and model with pre-trained weights
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = BertForSequenceClassification.from_pretrained('web/model', output_attentions = True)

device = torch.device('cpu')

model.to(device) 
model.eval()

@app.route('/predict', methods = ['POST'])
def predict():

    data = request.get_json()
    review_text = data.get('review', '')

    # Tokenize review and prepare input tensors
    inputs = tokenizer(review_text, return_tensors = 'pt', truncation = True, padding = True, max_length = 512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions

        # Calculate probabilities
        probs = torch.softmax(logits, dim = 1)
        confidence, prediction = torch.max(probs, dim = 1)
        
        sentiment = "positive" if prediction.item() == 1 else "negative"

        # Retrieve attention scores from the last layer
        attention_scores = attentions[-1].mean(dim = 1).squeeze(0).mean(dim = 0)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        token_scores = [{'token': token, 'attention': score.item()} for token, score in zip(tokens, attention_scores)]

    gc.collect()  # Clear memory

    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence.item(),
        'token_scores': token_scores
    })
