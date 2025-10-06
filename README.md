# ⚽ Soccer Match Predictor

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/accuracy-73%25-yellow.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/status-educational-orange.svg?style=for-the-badge)

**An Advanced Machine Learning System for Predicting La Liga Match Outcomes**

*Leveraging ensemble models and comprehensive feature engineering to forecast football match results*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-performance) • [Disclaimer](#-disclaimer)

---

</div>

## 📖 Overview

This project implements a sophisticated machine learning pipeline designed to predict the outcomes of La Liga football matches. By analyzing historical match data and employing state-of-the-art ensemble learning techniques, the system provides probabilistic predictions for match results.

> **⚠️ Important Notice**  
> This project is developed **exclusively for educational and research purposes**. It should **never be used for professional betting, gambling, or any commercial applications**. The predictions are probabilistic estimates based on historical data and do not guarantee future outcomes.

---

## ✨ Features

### 🤖 Advanced Machine Learning Models
- **Random Forest Classifier** - Ensemble decision tree model with robust performance
- **Gradient Boosting Classifier** - Sequential ensemble method achieving highest accuracy
- **LightGBM** - Efficient gradient boosting framework optimized for speed and performance

### 🔬 Comprehensive Feature Engineering
- **Team Form Analysis** - Weighted average of recent match performance
- **Head-to-Head Statistics** - Historical encounters between competing teams
- **Goal Analytics** - Average goals scored and conceded over recent matches
- **Shot Statistics** - Offensive and defensive shot patterns
- **Disciplinary Records** - Yellow and red card frequencies
- **Market Indicators** - Optional betting odds conversion to implied probabilities

### 💻 Interactive Command-Line Interface
- Real-time team selection and match setup
- Optional custom match statistics input
- Dynamic feature calculation and validation
- Confidence scores for all possible outcomes

### 📊 Transparent Predictions
- Detailed probability distributions for each outcome
- Model-specific predictions for comparison
- Confidence intervals and uncertainty quantification

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Soufiane-Tahiri/Soccer-Predictor.git
   cd SoccerMatchPredictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python src/PredictionTest.py --help
   ```

### Required Packages
- `scikit-learn` - Machine learning algorithms and utilities
- `lightgbm` - Gradient boosting framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `joblib` - Model serialization

---

## 📝 Usage

### Basic Usage

Run the predictor with your historical data:

```bash
python src/PredictionTest.py path/to/your/data.csv
```

If no CSV file is provided, the system will use default league averages:

```bash
python src/PredictionTest.py
```

### Interactive Prediction Flow

1. **Select Teams**
   - Choose home team from available options
   - Choose away team from remaining teams

2. **Enter Match Statistics** *(Optional)*
   - Recent form (last 5 matches)
   - Goal statistics
   - Head-to-head history
   - Shot averages
   - Card records
   - Betting odds

3. **Receive Predictions**
   - View probability distributions
   - Compare model predictions
   - Analyze confidence levels

### Example Output

```
╔══════════════════════════════════════════════════════════╗
║         MATCH PREDICTION RESULTS                         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  🏠 Real Madrid  vs  FC Barcelona 🛫                     ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📊 Model Predictions:                                   ║
║                                                          ║
║  Random Forest:                                          ║
║     🏆 Draw (57.0% confidence)                           ║
║                                                          ║
║  Gradient Boosting:                                      ║
║     🏆 Draw (56.6% confidence)                           ║
║                                                          ║
║  LightGBM:                                               ║
║     🛫 Away Win (47.7% confidence)                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Probability Distribution:
├─ Home Win:  35.4%  ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░
├─ Draw:      42.1%  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░
└─ Away Win:  22.5%  ▓▓▓▓▓░░░░░░░░░░░░░░░
```

---

## 📊 Performance Metrics

### Model Comparison

| Model                  | Training Accuracy | Validation Accuracy | F1-Score |
|------------------------|:-----------------:|:-------------------:|:--------:|
| Random Forest          | 68%               | 63%                 | 0.61     |
| **Gradient Boosting**  | **78%**           | **73%**             | **0.71** |
| LightGBM               | 72%               | 67%                 | 0.65     |

### Confusion Matrix Analysis

The Gradient Boosting model demonstrates the following performance on validation data:

- **Home Win Prediction:** 76% precision, 68% recall
- **Draw Prediction:** 64% precision, 71% recall
- **Away Win Prediction:** 81% precision, 77% recall

### Important Considerations

> 📌 **Key Points to Remember:**
> - Models achieve ~73% accuracy at best, meaning roughly 1 in 4 predictions may be incorrect
> - Football matches contain inherent unpredictability due to numerous factors
> - Historical performance does not guarantee future results
> - Model performance may vary across different seasons and leagues
> - Predictions should be interpreted as probabilistic estimates, not certainties

---

## 🎨 Feature Engineering Deep Dive

### Form Calculation
Weighted average of recent match results with exponential decay:
- **Win:** 3 points (weight: 1.0 for most recent)
- **Draw:** 1 point (weight: 0.8 for second most recent)
- **Loss:** 0 points (weight: 0.6, 0.4, 0.2 for older matches)

### Head-to-Head Metrics
- Win rate in last 10 encounters
- Average goal difference
- Home/away performance split

### Statistical Features
- **Goals:** Rolling average of goals scored and conceded
- **Shots:** Average shots on target and total shots
- **Cards:** Disciplinary record (yellows and reds per match)
- **Possession:** Average ball possession percentage *(if available)*

### Market Indicators
- Conversion of decimal odds to implied probabilities
- Bookmaker consensus analysis
- Overround adjustment for balanced probabilities

---

## ⚠️ Disclaimer

### Educational Purpose Only

This project is created **exclusively for educational and research purposes** to demonstrate:
- Machine learning concepts in sports analytics
- Feature engineering techniques
- Ensemble learning methods
- Data-driven prediction systems

### Limitations and Warnings

🚫 **DO NOT USE FOR:**
- Gambling or betting activities
- Professional sports predictions
- Commercial applications
- Financial decision-making

✅ **INTENDED FOR:**
- Learning machine learning concepts
- Academic research
- Personal experimentation
- Portfolio demonstration

### Prediction Accuracy

- Models are trained on historical La Liga data (seasons may vary)
- Predictions are probabilistic estimates, not guarantees
- Real-world match outcomes depend on numerous unpredictable factors
- Model performance may degrade over time without retraining
- Results may not generalize to other leagues or competitions

### Legal Notice

By using this software, you acknowledge that:
1. The predictions are for educational purposes only
2. The developers are not responsible for any losses incurred
3. You will not use this system for any form of gambling
4. You understand the probabilistic nature of predictions

---

## 📄 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2025 Soccer Match Predictor Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed modifications.

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation accordingly
- Maintain educational focus

---

## 📧 Contact & Support

For questions, suggestions, or discussions about this project:

- **Issues:** [GitHub Issues](https://github.com/Soufiane-Tahiri/Soccer-Predictor/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Soufiane-Tahiri/Soccer-Predictor/discussions)
---

## 🙏 Acknowledgments

- La Liga for historical match data
- Open-source machine learning community
- Contributors and testers who helped improve this project

---
