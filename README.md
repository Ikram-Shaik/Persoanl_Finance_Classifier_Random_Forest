# 📊 Personal Finance Classifier

A Streamlit-based machine learning web application that predicts a user’s financial personality — **Saver**, **Spender**, or **Neutral** — based on their income, expenses, lifestyle habits, and hobbies. This project uses a Random Forest classifier trained with clean, preprocessed financial data and is packaged using Docker for easy deployment.

---

## 🚀 Features

- 🔍 Predicts financial behavior based on user input
- 📊 Encodes hobbies and shopping frequency as behavioral features
- ⚙️ Model built using Random Forest with hyperparameter tuning
- 💾 Includes label encoding and scaling of numeric data
- 🖥️ Simple Streamlit web interface
- 🐳 Docker-ready for containerized deployment

---

## 🧠 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- **Docker**

---

## 📁 Project Structure

.
├── app.py                            # Streamlit frontend
├── model/                            # ML model files
│   ├── personal\_finance\_classifier\_model.pkl
│   ├── label\_encoder\_occupation.pkl
│   ├── label\_encoder\_city\_tier.pkl
│   ├── label\_encoder\_class.pkl
│   └── standard\_scaler.pkl
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker build instructions
├── README.md                         # Project documentation
└── .gitignore                        # Git ignored files

````

---

## 🧪 How to Run Locally

1. **Clone the Repository**:

```bash
git clone https://github.com/Ikram-Shaik/Persoanl_Finance_Classifier_Random_Forest.git
cd Persoanl_Finance_Classifier_Random_Forest
````

2. **Install Requirements**:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit App**:

```bash
streamlit run app.py
```

Then open your browser and visit: `http://localhost:8501`

---

## 🐳 Run with Docker

1. **Build the Docker Image**:

```bash
docker build -t finance-classifier-app .
```

2. **Run the Docker Container**:

```bash
docker run -p 8501:8501 finance-classifier-app
```

Then visit: `http://localhost:8501`

---

## 🧠 Model Details

* **Algorithm**: RandomForestClassifier
* **Accuracy**: \~95%
* **Preprocessing**:

  * Label encoding for `Occupation`, `City_Tier`, and `Class`
  * One-hot encoding for hobbies
  * Feature scaling for numerical columns
* **Target Classes**:

  * `Saver`
  * `Spender`
  * `Neutral`

---

## 📸 Live Link

🔗 [Hugging_Spaces](https://huggingface.co/spaces/Ikram-Shaik/Personal_Finance_Classifier)

---

## 📝 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for more details.

---

## 👤 Author

**Ikram Shaik**
🔗 [GitHub](https://github.com/Ikram-Shaik)
🔗 [LinkedIn](https://www.linkedin.com/in/ikram-shaik-/)

---

> Made with ❤️ using Python and Streamlit
