# Diabetes Prediction Project

## Overview
This project implements a machine learning system for predicting diabetes risk using multiple classification algorithms. The system uses various health parameters to predict whether a person is likely to have diabetes.

## Features
- Multiple ML models implementation (Random Forest, Naive Bayes, Bayesian Model)
- Interactive web interface for predictions
- Model performance comparison visualization
- Data preprocessing and scaling
- Cross-validation for model evaluation

## Project Structure
```
├── app.py                 # Flask web application
├── preprocess.py         # Data preprocessing scripts
├── train_models.py       # Model training scripts
├── data/                 # Dataset and processed data
├── models/              # Trained model files
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
└── visualizations/      # Model comparison plots
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/nilay2004/Diabetes-prediction.git
cd Diabetes-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Flask application:
```bash
python app.py
```
2. Open your web browser and navigate to `http://localhost:5000`
3. Enter the required health parameters to get diabetes prediction

## Models Used
- Random Forest Classifier
- Naive Bayes Classifier
- Bayesian Model

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS/JavaScript

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author
- Nilay Pandya
