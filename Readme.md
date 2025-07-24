# Air Quality Forecasting System


This project introduces an advanced Air Quality Forecasting System, seamlessly integrating Flask and state-of-the-art machine learning methodologies. Utilizing extensive historical air quality data and meteorological parameters, the system provides precise predictions of the Air Quality Index (AQI) for any selected city and date. Through an intuitive web interface, users can conveniently select their city, date, and preferred prediction model, enhancing their interaction with the system. The platform dynamically adapts available dates based on the chosen city, ensuring up-to-date and relevant predictions. Leveraging a diverse set of trained models, including Linear Regression, Random Forest Regression, XGBoost regressor, among others, the system optimizes prediction accuracy and reliability. Moreover, the final output page comprehensively displays various parameters alongside the AQI prediction, empowering users with comprehensive insights into air quality dynamics and potential health impacts. This system serves as a valuable tool for policymakers, environmentalists, and the general public, facilitating informed decision-making and fostering healthier communities in the face of air pollution challenges.


## 🔍 Features

- 📅 Select city and date for AQI forecasts
- 🤖 Choose your model: Linear Regression, Random Forest, or XGBoost
- 🌫️ Predict pollutants: PM2.5, PM10, NO₂, CO, SO₂, O₃, etc.
- 📈 Interactive web interface built using Flask & Jinja templates



## 🛠️ Tech Stack

### 👨‍💻 Frontend
- **HTML5**, **CSS3**, **Bootstrap** – Clean and responsive UI
- **Jinja2** – Templating engine for Flask

### 🧠 Backend
- **Python 3.7+**
- **Flask** – Micro web framework for routing and user interaction
- **pandas**, **numpy** – Data handling and preprocessing
- **scikit-learn** – ML model training and evaluation
- **XGBoost** – Advanced gradient boosting model

### 🗄️ ML Models
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

### 🔧 Tools & Libraries
- **Matplotlib / Seaborn** (for data visualization in notebooks)
- **Jupyter Notebook** (for model development & EDA)
- **pickle** – For model serialization
- **virtualenv** – (Recommended) for dependency management


## 🚀 Getting Started

```bash
git clone https://github.com/Arya920/AirQualityPredictionFlask.git
cd AirQualityPredictionFlask
python -m venv env
source env/bin/activate        # On Windows: env\Scripts\activate
pip install -r requirements.txt
python flask_app.py

Through its intuitive web interface, users can seamlessly select their city of interest, specify desired dates, and even choose their preferred prediction model, improving accessibility and user engagement. Beyond its technical capabilities, this project underscores the critical importance of air quality forecasting in protecting public health, preserving environmental integrity, and fostering sustainable development. By empowering policymakers, environmental advocates, researchers, and the general public with actionable information, the system catalyzes proactive interventions to mitigate the adverse effects of air pollution and promote healthier communities. Through this collaborative effort bridging technology and environmental science, we aspire to pave the way for a cleaner, greener future where the air we breathe is safe, sustainable, and conducive to well-being.

