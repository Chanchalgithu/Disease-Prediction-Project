# 🏥 Disease Prediction in Healthcare

A **Machine Learning-powered Flask web application** that predicts diseases based on patient symptoms. Developed for healthcare assistance and early medical consultation insights.
    
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🎯 Problem Statement

Develop a robust ML classification system that:
- Predicts diseases accurately based on **132 symptoms**
- Classifies into **41 different diseases**
- Provides early insights for medical consultation
- Maintains prediction logs for analysis

---

## ✨ Key Features

🔍 **Smart Prediction Engine**
- Symptom-based disease classification using trained ML models
- High accuracy prediction with confidence scoring

🌐 **Interactive Web Interface**
- User-friendly Flask web application
- Intuitive symptom selection interface
- Real-time prediction results

📊 **Data Management**
- Automatic prediction logging in CSV format
- JSON API endpoints for system integration
- Comprehensive data preprocessing pipeline

☁️ **Cloud Deployment**
- Production-ready deployment on Render
- Scalable and accessible from anywhere

---
## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Render, AWS, Docker-ready |
| **Development** | Jupyter Notebook, VS Code |

---
## 📁 Project Architecture

```
DISEASE_PREDICTION/
├── data/
│   ├── processed/
│   │   ├── feature_columns.json
│   │   ├── label_mapping.json
│   │   ├── test_clean.csv
│   │   └── train_clean.csv
│   └── raw/
│       ├── Testing.csv
│       └── Training.csv
├── models/
│   └── disease_model.pkl
├── Notebooks/
│   ├── data_preparation.ipynb
│   ├── exploratory_data_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── __init__.py
│   └── app.py
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   └── result.html
├── predictions.csv
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chanchalgithu/Disease-Prediction-Project.git
   cd disease-prediction
   ```

       
2. **Install dependencies**
  
   pip install -r requirements.txt

3. **Run the application**
 
   python src/app.py
 
4. **Access the application**
   - Open your browser and navigate to `http://127.0.0.1:5000`

---

## 💡 How It Works

### User Workflow
1. **Symptom Selection** → Choose symptoms from comprehensive dropdown menu
2. **Prediction** → ML model processes symptoms and predicts disease
3. **Results** → View predicted disease with confidence score
4. **Logging** → Prediction automatically saved to `predictions.csv`

### API Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | Health check | `{"status": "ok"}` |
| `/predict` | POST | Disease prediction | `{"prediction": "Disease Name"}` |

**API Usage Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 1, 0, 0, 1, 0, ...]}'
```

---
# 📊Model Performance

 - **Algorithms Tried:** Logistic Regression, Naive Bayes, SVM (RBF), Random Forest, XGBoost
 - **Training Data:** 132 symptoms × 41 diseases
 - **Dataset Size:** 4920 training samples, 42 test samples
 - **Preprocessing:** Missing value handling, label encoding, scaling, class balancing
 - **Validation Accuracy (Cross-Validation):** 100% ± 0.00 (for all models)
 - **Final Model Accuracy (Random Forest):** 97.62% on test dataset
 - **Classification Report:** Precision, Recall, F1-score ~0.98–0.99 (weighted average)
 - **Note:** Slightly lower recall (0.50) observed for Fungal Infection due to very few test samples
 - **Prediction Time:** < 100ms per request       
---

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=production
PORT=5000
MODEL_PATH=models/disease_model.pkl
```

### Deployment Settings
- **Production Server:** Gunicorn
- **Host:** `0.0.0.0`
- **Port:** Dynamic (Render) or `5000` (local)

---

# 📚 Dataset Information

 - **Source:** Medical Dataset
 - **Training Records:** 4,920 samples
 - **Testing Records:** 42 samples
 - **Features:** 132 binary symptom indicators
 - **Target Classes:** 41 unique diseases (prognosis column)
 - **Balance:** Dataset mostly balanced, with both common (Fungal infection, Allergy) and rare (AIDS, Alcoholic Hepatitis) diseases
   
---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PWSkills** for the Mini-Hackathon opportunity
- **scikit-learn** community for excellent ML tools
- **Flask** team for the lightweight web framework
- **Render** for reliable cloud hosting

---

## 📞 Support

For questions and support:
- 📧 Email: chanchalraikwar447@gmail.com
- 💬 Issues: [GitHub Issues](https://github.com/Chanchalgithu/Disease-Prediction-Project/issues)
- 🌟 Give us a star if this project helped you!

---

<div align="center">

**Built with ❤️ for better healthcare accessibility**

[Live Demo](https://your-app.render.com) • [Documentation](docs/) • [Report Bug](https://github.com/Chanchalgithu/Disease-Prediction-Projec/issues)

</div>