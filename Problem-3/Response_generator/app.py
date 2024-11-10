from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch 


# Initialize the Flask app
app = Flask(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
classifier = pipeline("zero-shot-classification", model = 'facebook/bart-large-mnli',device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model directly


tokenizer = AutoTokenizer.from_pretrained("Yihui/t5-small-text-summary-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("Yihui/t5-small-text-summary-generation")
model = model.to(device)

# Add padding token to tokenizer if it does not exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings to accommodate the new padding token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# Define classes for classification
categories = ["Meeting Request", "Status Update", "General Query"]

def summarize_email_thread(thread_text):
    # Tokenize the input text
    inputs = tokenizer(thread_text, return_tensors="pt",truncation = True)
    # inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate summary using model.generate()
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length= 1012,
        min_length=50,
        num_beams=4,
        early_stopping=False,
        no_repeat_ngram_size=10,
    )
    
    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        email_content = request.form.get("email_content")
        if email_content:
            # Sample email content (ensure it's defined and not too large

            # Classify the email
            print("Starting classification...")
            result = classifier(email_content, candidate_labels=categories, multi_label=False,truncation=True)
            print("Classification complete.")
            result = {result['labels'][0]}
            print("Classification result:", result)

            prompt_prefix = f"Generate a response mail to the following email in {result} tone "
            response = summarize_email_thread(prompt_prefix + email_content)
            # prompt = f"{prompt_prefix} {email_content}"
            # print("Starting text generation...")
            # responses = generator(prompt, max_length=513, num_return_sequences=1,truncation=True,pad_token_id=generator.tokenizer.eos_token_id)  # Adjust max_length
            # print("Text generation complete.")
            # response = responses[0]['generated_text'][len(prompt_prefix):]

    
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)

