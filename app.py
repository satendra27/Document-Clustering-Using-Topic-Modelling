from flask import Flask, request, render_template
import pickle
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from flask_mail import Mail, Message

nltk.download('stopwords')

app = Flask(__name__)

# Load saved models
vectorizer = pickle.load(open('Vectorizer.pkl', 'rb'))
kmeans = pickle.load(open('KMeans-clustering.pkl', 'rb'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = re.sub(r'\b\w{1,2}\b|\b\w{13,}\b', '', text)
    text = re.sub(r'\d+', '', text)
    frequent_words = set(['article', 'email', 'write', 'writes', 'wrote', 'subject', 're'])
    text = ' '.join(word for word in text.split() if word.lower() not in frequent_words)
    return text


cluster_to_category = {0: '🏍️rec.motorcycles',
 1: '🌍talk.politics.mideast',
 2: '🔐sci.crypt',
 3: '🏥sci.med',
 4: 'comp.sys.mac.hardware',
 5: '🏑rec.sport.hockey',
 6: '🛒misc.forsale',
 7: 'comp.graphics',
 8: '🪟comp.os.ms-windows.misc',
 9: 'rec.motorcycles',
 10: '💻comp.os.ms-windows.misc',
 11: '⚾rec.sport.baseball',
 12: 'talk.politics.mideast',
 13: '🚗rec.autos',
 14: '🍎comp.sys.mac.hardware',
 15: '📈comp.graphics',
 16: '🔫talk.politics.guns',
 17: 'rec.sport.hockey',
 18: '✝️soc.religion.christian',
 19: '🚀sci.space'}


def get_top_keywords(kmeans, vectorizer, cluster_id, n_terms=10):
    centroids = kmeans.cluster_centers_
    terms = vectorizer.get_feature_names_out()
    top_indices = centroids[cluster_id].argsort()[::-1][:n_terms]
    return [terms[i] for i in top_indices]


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'baghelsatendra27@gmail.com'          # your Gmail
app.config['MAIL_PASSWORD'] = 'Satya.8956'             # use App Password, not real password
app.config['MAIL_DEFAULT_SENDER'] = 'baghelsatendra27@gmail.com'

mail = Mail(app)


prediction_results = []
@app.route('/', methods=['GET', 'POST'])
def index():
    cluster = None
    category = None
    top_words = []
    success = None
    error = None

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        # 🟦 Prediction Form
        if form_type == 'prediction':
            uploaded_file = request.files.get('file')
            input_text = request.form.get('text')

            if uploaded_file and uploaded_file.filename.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                texts = [content]
            elif input_text.strip():
                texts = [input_text.strip()]
            else:
                error = "Please enter text or upload a file."
                return render_template("index.html", error=error)

            for text in texts:
                clean_text = preprocess(text)
                vector = vectorizer.transform([clean_text])
                cluster = kmeans.predict(vector)[0]
                category = cluster_to_category.get(cluster, "Unknown")
                top_words = get_top_keywords(kmeans, vectorizer, cluster)

                # Wordcloud
                words = " ".join(top_words)
                wordcloud = WordCloud(width=1000, height=500, background_color=None,mode='RGBA', colormap='viridis').generate(words)
                plt.figure(figsize=(8, 6))
                plt.title("Top Keywords in the Cluster")
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.savefig(r"static/top_word_wordcloud.png",transparent=True)  # Save locally in static/

                prediction_results.append((text[:100] + "...", cluster, category))

        # 🟩 Contact Form
        elif form_type == 'contact':
            name = request.form['name']
            sender_email = request.form['email']
            message_body = request.form['message']

            msg = Message(subject=f"New Contact Form Submission from {name}",
                          sender=sender_email,
                          recipients=['baghelsatendra27@gmail.com'],
                          body=f"Name: {name}\nEmail: {sender_email}\n\nMessage:\n{message_body}")
            try:
                mail.send(msg)
                success = True
            except Exception as e:
                print("Error sending email:", e)
                success = False

    return render_template('index.html',
                           cluster=cluster,
                           category=category,
                           top_words=top_words,
                           predictions=prediction_results,
                           success=success,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)