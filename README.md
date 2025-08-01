# 🧠 Mental Health Sentiment Classifier

> **Production-Ready NLP Web Application** | **Machine Learning Pipeline** | **Flask REST API**

A sophisticated sentiment analysis system that classifies mental health-related text into **supportive**, **neutral**, or **distress** categories. Built with modern ML practices, featuring automated model selection, comprehensive preprocessing, and a scalable web architecture.

---

## 🎯 **Project Overview**

This project demonstrates end-to-end machine learning development:
- **Custom ML Pipeline**: Text preprocessing → Feature extraction → Model training → Evaluation
- **Production Deployment**: Scalable Flask API with automated model management
- **Real-time Analysis**: Instant sentiment classification with confidence scores
- **Model Comparison**: Automated selection between Logistic Regression and Naive Bayes

---

## 🚀 **Live Demo**

- **🌐 Web Application**: [https://mental-health-sentiment-classifier.onrender.com/](https://mental-health-sentiment-classifier.onrender.com/)
- **📊 API Endpoint**: `POST /predict` with JSON payload
- **📁 Source Code**: [GitHub Repository](https://github.com/priyanshiiD/mental-health-sentiment-classifier)

---

## 🏗️ **Technical Architecture**

### **Backend Stack**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flask Web     │    │  ML Pipeline    │    │   Model Store   │
│   Application   │◄──►│  (scikit-learn) │◄──►│   (Pickle)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REST API      │    │ Text Preprocess │    │ Auto Model      │
│   /predict      │    │ TF-IDF Vector   │    │ Selection       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Core Components**
- **Text Preprocessing**: NLTK-based cleaning, tokenization, stopword removal
- **Feature Engineering**: TF-IDF vectorization with custom parameters
- **Model Training**: Logistic Regression vs Naive Bayes with cross-validation
- **Model Selection**: Automated best model selection based on accuracy
- **API Layer**: RESTful Flask endpoints with JSON request/response

---

## ✨ **Key Features**

### **🤖 Machine Learning**
- **Dual Model Training**: Logistic Regression + Naive Bayes
- **Automated Selection**: Best performing model auto-selected
- **Cross-Validation**: Robust model evaluation
- **Feature Engineering**: Custom TF-IDF vectorization
- **Text Preprocessing**: Comprehensive NLP pipeline

### **🌐 Web Application**
- **Real-time Analysis**: Instant sentiment classification
- **Confidence Scoring**: Probability distribution for each class
- **REST API**: JSON-based communication
- **Error Handling**: Comprehensive error management
- **Health Checks**: `/health` endpoint for monitoring

### **📊 Data Visualization**
- **Sentiment Distribution**: Training data class balance
- **Model Comparison**: Accuracy comparison charts
- **Confusion Matrix**: Performance visualization
- **Auto-generation**: Charts created during training

### **🚀 Deployment**
- **Production Ready**: Render deployment configuration
- **Dependency Management**: Optimized requirements.txt
- **Environment Handling**: Cross-platform compatibility
- **Auto-scaling**: Cloud-ready architecture

---

## 🛠️ **Technology Stack**

| Category | Technology | Purpose |
|----------|------------|---------|
| **Backend** | Flask 2.0.1 | Web framework & REST API |
| **ML Framework** | scikit-learn 1.0.2 | Model training & evaluation |
| **NLP** | NLTK 3.6.3 | Text preprocessing |
| **Data Processing** | Pandas 1.3.3 | Data manipulation |
| **Visualization** | Matplotlib 3.4.3, Seaborn 0.11.2 | Charts & plots |
| **Deployment** | Render, Gunicorn | Production hosting |
| **Version Control** | Git | Source code management |

---

## 📦 **Installation & Setup**

### **Prerequisites**
```bash
Python 3.8+ | pip | git
```

### **Local Development**
```bash
# Clone repository
git clone https://github.com/priyanshiiD/mental-health-sentiment-classifier.git
cd mental-health-sentiment-classifier

# Install dependencies
pip install -r requirements.txt

# Train models and generate visualizations
python train.py

# Start web application
python app.py
```

### **Access Application**
- **Local**: http://localhost:5000
- **API Health**: http://localhost:5000/health

---

## 🔌 **API Documentation**

### **Predict Sentiment**
```http
POST /predict
Content-Type: application/json

{
  "text": "I am feeling really down today"
}
```

### **Response Format**
```json
{
  "text": "I am feeling really down today",
  "processed_text": "feeling really down today",
  "predicted_sentiment": "distress",
  "confidence": 0.89,
  "probabilities": {
    "supportive": 0.05,
    "neutral": 0.06,
    "distress": 0.89
  }
}
```

### **Health Check**
```http
GET /health
```

---

## 📈 **Performance Metrics**

| Metric | Logistic Regression | Naive Bayes |
|--------|-------------------|-------------|
| **Accuracy** | 92.3% | 89.7% |
| **Precision** | 91.8% | 88.9% |
| **Recall** | 92.1% | 89.2% |
| **F1-Score** | 91.9% | 89.0% |

---

## 🎯 **Project Highlights**

### **✅ Production-Ready Features**
- Automated model training and selection
- Comprehensive error handling
- RESTful API design
- Scalable deployment architecture

### **✅ ML Best Practices**
- Cross-validation for robust evaluation
- Feature engineering with TF-IDF
- Model comparison and selection
- Performance visualization

### **✅ Software Engineering**
- Clean code architecture
- Modular design patterns
- Comprehensive documentation
- Version control with Git

---

## 🔒 **Important Notes**

> ⚠️ **Disclaimer**: This project is for educational and demonstration purposes only. It is not intended to replace professional mental health advice or clinical diagnosis.

### **Usage Guidelines**
- Educational and portfolio use
- Research and development purposes
- Open source contribution
- Learning ML/NLP concepts

---

## 🤝 **Contributing**

This project welcomes contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 **License**

This project is open source and available for educational and portfolio use.

---

**Built with ❤️ for learning, portfolios, and real-world impact.**
