
# Document Clustering Using Topic Modelling

## Objective

The objective of this project is to develop an interactive web application that automatically analyzes and categorizes textual documents using unsupervised machine learning techniques, specifically KMeans clustering. By extracting and grouping documents based on similar topics, the system aims to:

- #### Provide a visual and intuitive understanding of underlying themes using topic modeling.

- #### Assist users in identifying the most relevant category for their text.

- #### Display top keywords and generate a wordcloud for easy interpretation.

- #### Show cluster probability distributions to reflect the model’s confidence in topic assignments.

 
## 🚀 Features

- 🧠 **Topic Modeling using KMeans Clustering**
- 📄 Supports input via form or `.txt` file
- ☁️ **WordCloud generation** for each cluster
- 📊 **Bar chart** of probability distribution across all clusters
- 🎨 Responsive and modern UI with animations and interactive sections
- 📁 File upload and preprocessing pipeline
- 📬 Deployment-ready for platforms like Render etc.

## 🛠️ Tech Stack

- `Python`
- `Flask`
- `Scikit-learn`
- `NLTK`
- `Pandas`
- `Numpy`
- `Seaborn`
- `Matplotlib`
- `WordCloud`
- `Machine learning`
- `HTML`, `CSS`, `Bootstrap`

---

# 🧠 Document Clustering Using Topic Modeling (KMeans)

An interactive web application that classifies uploaded or pasted text into topic-based clusters using **KMeans clustering**, with support for visualizations like **word clouds** and **probability distributions**.

---

## 📸 Screenshots

### 🏠 Home Page
![Home Screenshot](Project%20Snapshots/Screenshot%202025-07-07%20231822.png)

### 📊 Prediction Section
![Prediction Result](Project%20Snapshots/Screenshot%202025-07-07%20233143.png)

### 🔢 Cluster Probability Chart
![Cluster Probability](Project%20Snapshots/Screenshot%202025-07-08%20112950.png)

> 📝 Make sure these images are placed in your `/static/screenshots/` folder or use relative GitHub raw URLs if hosted there.

---
🚀 Project Functionalities
This web application enables users to discover the category or topic of a document using unsupervised machine learning. The main features include:

🧠 1. Prediction Module
Accepts input via text box or .txt file upload.

Applies text preprocessing and TF-IDF vectorization.

Uses KMeans clustering to assign the input document to a cluster.

Displays the predicted topic/category using a clean UI.

📊 2. Probability Distribution
After prediction, the model computes the probability distribution across all clusters.

Visualized using a bar chart, highlighting the predicted cluster.

Helps users understand how confident the model is in its prediction.

📂 3. Recent Predictions
Shows a short summary (first 100 characters) of recently analyzed documents.

Displays the corresponding cluster number and predicted topic for each.

Allows users to track and compare previous results during the session.

📈 4. Analysis Dashboard
Generates a WordCloud of the top 10 keywords from the predicted cluster.

Helps visualize the most frequent and significant terms in the topic.

Enhances interpretability of the clustering results.

📌 5. About Section
Describes the purpose of the project and the technologies used.

Outlines the advantages of using unsupervised topic modeling in real-world applications.

📫 6. Contact Form & Footer
Users can send feedback or inquiries directly via the Contact Us form.

The footer includes useful links (GitHub repo, deployment link, credits, etc.).


## 🌐 Live Demo

👉 [Visit the deployed app](https://document-clustering-using-topic-modelling.onrender.com)

---

## 📦 GitHub Repository

🔗 [GitHub Repo](https://github.com/satendra27/Document-Clustering-Using-Topic-Modelling.git)

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/document-clustering.git
cd document-clustering
pip install -r requirements.txt
python app.py


