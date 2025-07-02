# Electric Vehicle Cost Analysis & Prediction

Welcome! This project is focused on analyzing and predicting the costs of electric vehicles (EVs) using machine learning. It provides a web-based interface for data exploration, user management, and EV price prediction, aiming to make EV data accessible and actionable.

## Problem Statement

With the growing adoption of electric vehicles, understanding the factors that influence their cost is crucial for both consumers and manufacturers. However, the complexity and variety of EV features make price prediction challenging. This project addresses this by leveraging machine learning to predict EV prices based on technical and categorical features.

## Solution

- **Data Analysis:** The app provides visualizations and statistics to help users understand the EV dataset.
- **User Management:** Users can register, log in, and manage their profiles.
- **Price Prediction:** Users can input EV specifications and receive a predicted price using a trained Random Forest model.
- **Accessible Interface:** Built with Flask, the app offers a simple web UI for all features.

## Technologies Used

- **Frontend:** HTML, CSS (with Flask templates)
- **Backend:** Python, Flask
- **Database:** SQLite3
- **Machine Learning:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Other:** Jupyter Notebooks for data preprocessing

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <project-folder>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On Unix/Mac
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   python create_table.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```
   The app will be available at [http://localhost:8000](http://localhost:8000).

## Usage

- **Register/Login:** Create a user account or log in.
- **Explore Dataset:** View and analyze the EV dataset.
- **Train Model:** Train the machine learning model and view accuracy metrics.
- **Predict Price:** Enter EV specifications to get a price prediction.

## Contributions

We welcome contributions! Please follow these steps:

1. **Understand the project requirements.**
2. **Create an issue** to report bugs or suggest features.
3. **Fork the repository** and create a new branch for your changes.
4. **Submit a pull request** with a clear description of your changes.



---


**Contact:** [Resham Sai Pranathi/pranathi9191@gmail.com]
