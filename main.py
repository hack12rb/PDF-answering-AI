from flask import Flask, render_template, jsonify, request
import os
import pathlib

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = os.path.join(pathlib.Path(__file__).parent, "uploads")

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


def get_answer(pdf_path: str, question: str):
    import fitz
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import nltk
    from gensim.models import Word2Vec
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        words = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    def get_embedding(text, model):
        words = preprocess_text(text)
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if not word_vectors:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    def find_best_answer(question, context, model):
        question_embedding = get_embedding(question, model)
        context_sentences = context.split(".")
        best_similarity = -1
        best_sentence = None
        for sentence in context_sentences:
            sentence_embedding = get_embedding(sentence, model)
            similarity = cosine_similarity([question_embedding], [sentence_embedding])[
                0
            ][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = sentence
        return best_sentence

    pdf_text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(pdf_text)
    model = Word2Vec(
        [preprocessed_text], vector_size=100, window=5, min_count=1, workers=4
    )

    answer = find_best_answer(question, pdf_text, model)
    return answer


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/questions")
def questions():
    return render_template("questions.html")


@app.route("/answer", methods=["POST"])
def answers():
    data = request.form
    question = data.get("question")
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "sample.pdf")
    answer = get_answer(pdf_path, question)
    return jsonify({
        "status": "success",
        "message": "Answer found successfully",
        "answer": answer
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "failed", "message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "failed", "message": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "sample.pdf")
        file.save(filepath)
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "File uploaded successfully",
                    "filename": file.filename,
                }
            ),
            200,
        )


if __name__ == "__main__":
    app.run(debug=True)
