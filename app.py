import os
import pickle
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Built-in modern glassmorphism UI template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .container {
            width: 100%;
            max-width: 680px;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid rgba(99, 102, 241, 0.3);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(to right, #f8fafc, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        p.subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 16px;
            color: #f8fafc;
            font-size: 1rem;
            resize: vertical;
            outline: none;
            transition: all 0.2s ease;
        }

        textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .actions {
            margin-top: 16px;
            display: flex;
            gap: 12px;
        }

        button {
            flex: 1;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #fff;
            border: none;
            border-radius: 12px;
            padding: 14px 20px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        }

        button:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 28px;
            padding: 20px;
            border-radius: 16px;
            animation: fadeIn 0.3s ease;
        }

        .result-positive {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
        }

        .result-negative {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
        }

        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }

        .confidence {
            font-size: 0.85rem;
            color: #cbd5e1;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">NLP Model Engine</span>
            <h1>Sentiment Analyzer</h1>
            <p class="subtitle">Enter any text or review below to predict the emotional tone.</p>
        </div>

        <form method="POST" action="/">
            <textarea name="text" placeholder="Type or paste text here (e.g., 'The product quality is absolutely wonderful!')..." required>{{ user_text or '' }}</textarea>
            <div class="actions">
                <button type="submit">Analyze Sentiment</button>
            </div>
        </form>

        {% if sentiment %}
        <div class="result-card result-{{ sentiment.lower() }}">
            <div class="result-title">
                {% if sentiment.lower() == 'positive' %}
                    <span>😊 Positive Sentiment</span>
                {% else %}
                    <span>😞 Negative Sentiment</span>
                {% endif %}
            </div>
            <div class="confidence">Confidence Score: <strong>{{ confidence }}%</strong></div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    confidence = None
    user_text = ""

    if request.method == 'POST':
        user_text = request.form.get('text', '').strip()
        if user_text:
            text_vec = vectorizer.transform([user_text])
            prediction = model.predict(text_vec)[0]
            
            # Extract probability confidence if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(text_vec)[0]
                pred_idx = list(model.classes_).index(prediction)
                confidence = round(probs[pred_idx] * 100, 2)
            else:
                confidence = 100.0

            sentiment = str(prediction).capitalize()

    return render_template_string(
        HTML_TEMPLATE,
        sentiment=sentiment,
        confidence=confidence,
        user_text=user_text
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
