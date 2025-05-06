import torch
from flask import Flask, request, render_template_string, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from enum import Enum
from pyngrok import ngrok

app = Flask(__name__)

class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

tokenizer = AutoTokenizer.from_pretrained(
    "Swathi8378/tinyllama_Text2Sql_lora",
    pad_token=ChatmlSpecialTokens.pad_token.value,
    bos_token=ChatmlSpecialTokens.bos_token.value,
    eos_token=ChatmlSpecialTokens.eos_token.value,
    additional_special_tokens=ChatmlSpecialTokens.list(),
    trust_remote_code=True
)
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(
    base_model,
    "Swathi8378/tinyllama_Text2Sql_lora",
    torch_dtype=torch.float16
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def construct_prompt(schema: str, question: str) -> str:
    return (
        f"{ChatmlSpecialTokens.system.value}\n"
        f"You are a SQL assistant. Use the following schema to answer queries.\n{schema}\n{ChatmlSpecialTokens.eos_token.value}\n"
        f"{ChatmlSpecialTokens.user.value}\n"
        f"{question}\n{ChatmlSpecialTokens.eos_token.value}\n\n\n"
        f"{ChatmlSpecialTokens.assistant.value}"
    )

def generate_sql(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = tokenizer.batch_decode(outputs)
    outputs = [output.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip() for output in outputs]
    return outputs

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Text-to-SQL Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f7f7f7;
        }
        .container {
            max-width: 700px;
            margin: 50px auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        textarea {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
            resize: vertical;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .box {
            text-align: left;
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            white-space: pre-wrap;
        }
        .label {
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🧠 Text-to-SQL Generator</h2>
        <form action="/generate" method="post">
            <label>🗂️ Schema:</label><br>
            <textarea name="schema" rows="6" placeholder="Enter your table definitions here...">{{ schema }}</textarea><br><br>
            <label>❓ Question:</label><br>
            <textarea name="question" rows="4" placeholder="Enter a natural language question...">{{ question }}</textarea><br><br>
            <button type="submit">Generate SQL</button>
        </form>

        {% if result %}
            <div class="box">
                <div class="label">🗂️ Schema:</div>
                {{ schema }}
            </div>
            <div class="box">
                <div class="label">❓ Question:</div>
                {{ question }}
            </div>
            <div class="box">
                <div class="label">📝 Generated SQL:</div>
                {{ result }}
            </div>
        {% endif %}
    </div>
</body>
</html>
'''



@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    schema = request.form.get('schema', '').strip()
    question = request.form.get('question', '').strip()
    if not schema or not question:
        return render_template_string(HTML_TEMPLATE, result="Please provide both schema and question.")
    prompt = construct_prompt(schema, question)
    sql = generate_sql(prompt)[0].split(';')[0] + ';'
    return render_template_string(HTML_TEMPLATE, schema= schema, question=question, result=sql)

public_url = ngrok.connect(5000)
print(" * ngrok tunnel: ", public_url)
if __name__ == '__main__':
    app.run()
