# 🏥 AI Diagnostic Tool Adoption Dashboard - Singapore Hospitals

An interactive Streamlit dashboard analyzing physician adoption patterns of AI diagnostic tools in Singapore hospitals, based on the Technology Acceptance Model (TAM).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Dashboard Pages](#dashboard-pages)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This dashboard provides comprehensive analytics on physician attitudes toward AI diagnostic tools across Singapore hospitals. It uses synthetic data based on the Technology Acceptance Model (TAM) framework to demonstrate:

- **Perceived Usefulness (PU)**: How useful physicians find AI diagnostic tools
- **Ease of Use (EOU)**: How easy physicians find AI tools to use
- **Trust**: Level of trust in AI diagnostic recommendations
- **Intention to Adopt (ITA)**: Likelihood of adopting AI tools in practice

## ✨ Features

- 📊 **Interactive KPI Dashboard**: Real-time metrics and score distributions
- 🔗 **Correlation Analysis**: Heatmap visualization of TAM construct relationships
- 📈 **Demographic Breakdowns**: Scores by specialty, hospital size, age group
- 🎯 **Scatter Plot Analysis**: Explore relationships between variables
- 🤖 **OLS Prediction Model**: Predict adoption intention with interactive inputs
- 🔍 **Advanced Filtering**: Filter by age, specialty, and hospital size

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-adoption-dashboard.git
cd ai-adoption-dashboard
Create virtual environment (recommended)
Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
Bash

pip install -r requirements.txt
Generate synthetic data
Bash

python generate_data.py
Run the dashboard
Bash

streamlit run app.py
The dashboard will open in your browser at http://localhost:8501

📖 Usage
Generating Data
Run the data generator to create the synthetic dataset:

Bash

python generate_data.py
This creates adoption_data.csv with 500 physician records containing:

Demographics (age, specialty, hospital size, experience)
TAM scores (PU, EOU, Trust, ITA)
Derived features (adoption intent, age groups)
Running the Dashboard
Bash

streamlit run app.py
Using Filters
Use the sidebar to filter data by:

Age range (slider)
Medical specialty (multiselect)
Hospital size (multiselect)
All visualizations update automatically based on filters

📊 Data Description
Column	Description	Range/Values
physician_id	Unique identifier	PH0001-PH0500
age	Physician age	25-65 years
specialty	Medical specialty	Radiology, Oncology, Cardiology, General Medicine, Surgery
hospital_size	Hospital bed capacity	Small (<200), Medium (200-500), Large (>500)
pu_score	Perceived Usefulness	1-5 (mean: 3.5)
eou_score	Ease of Use	1-5 (mean: 3.2, corr with age: -0.3)
trust_score	Trust in AI	1-5 (mean: 3.0)
ita_score	Intention to Adopt	1-5 (regressed on PU+EOU+Trust)
📑 Dashboard Pages
1. Overview
Key performance indicators (average scores, adoption rate)
Score distribution histograms
Summary statistics table
2. Correlations
Interactive correlation heatmap
Detailed correlation analysis with ITA
3. Demographics
Bar charts by demographic categories
Specialty and hospital size breakdowns
Age group analysis with box plots
4. Scatter Analysis
Customizable scatter plots with trend lines
PU vs ITA bubble chart
Quadrant analysis
5. Prediction Model
OLS regression results (ITA ~ PU + EOU + Trust)
Model coefficients visualization
Actual vs Predicted comparison
Interactive prediction tool
🌐 Deployment
Deploy to Streamlit Cloud
Push your code to GitHub

Go to share.streamlit.io

Click "New app" and connect your repository

Set the main file path to app.py

Click "Deploy"

Environment Variables
No environment variables required for basic deployment.

🛠️ Tech Stack
Frontend: Streamlit
Data Processing: Pandas, NumPy
Visualization: Plotly
Machine Learning: Scikit-learn
Styling: Custom CSS
📁 Project Structure
ai-adoption-dashboard/
├── app.py                 # Main Streamlit application
├── generate_data.py       # Synthetic data generator
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── adoption_data.csv     # Generated dataset (after running generate_data.py)
└── .gitignore           # Git ignore file
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
Created for demonstration purposes.

🙏 Acknowledgments
Technology Acceptance Model (TAM) framework by Davis (1989)
Streamlit team for the amazing framework
Plotly for interactive visualizations
Made with ❤️ using Streamlit