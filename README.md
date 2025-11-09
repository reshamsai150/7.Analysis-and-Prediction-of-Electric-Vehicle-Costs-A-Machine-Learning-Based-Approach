# 🚗 Analysis and Prediction of Electric Vehicle Costs

![GitHub License](https://img.shields.io/github/license/reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach)
![GitHub last commit](https://img.shields.io/github/last-commit/reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach)
![GitHub issues](https://img.shields.io/github/issues/reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach)
![GitHub stars](https://img.shields.io/github/stars/reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach)


# ⚡ Electric Vehicle Cost Analysis & Prediction

A lightweight, open-source web application for analyzing and predicting the cost of electric vehicles (EVs) using machine learning. This app simplifies EV data exploration, provides user management, and predicts EV prices based on technical specifications.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Built%20with-Python-blue)


---

## 📑 Table of Contents
- 🚀 Features
- 📊 How to Use
- 🛠️ Tech Stack
- ⚙️ Getting Started
- 📁 Project Structure
- 🧑‍💻 Contributing
- 💡 Feature Ideas & Roadmap
- 📄 License
- 📬 Contact

---

## 🚀 Features
- 📈 **Data Analysis Dashboard** – Explore the EV dataset with interactive graphs and statistics.
- 🧑‍💼 **User Authentication** – Register, login, and manage user sessions.
- 🤖 **Price Prediction Engine** – Predict EV prices based on specifications using a trained Random Forest model.
- 🖥️ **Simple Web Interface** – Built with Flask for accessible usage.
- 📂 **Data Preprocessing** – Performed using Jupyter Notebooks and visualized with seaborn & matplotlib.

---

## 📊 How to Use
1. 🔐 **Register/Login** – Create an account or log in as an existing user.
2. 📁 **Explore Dataset** – Access visual insights into the EV dataset.
3. 🧠 **Train Model** – Retrain the ML model and view accuracy.
4. 📤 **Predict EV Price** – Enter specifications to receive an instant price estimate.

---

## 🛠️ Tech Stack

**Frontend**
- HTML, CSS (Jinja2 templates via Flask)

**Backend**
- Python, Flask

**Machine Learning**
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

**Database**
- SQLite3

**Other Tools**
- Jupyter Notebooks (for analysis & preprocessing)

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.x installed

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach.git
cd ev-price-predictor

# Create and activate a virtual environment
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up the database
python create_table.py

# Run the app
python app.py
```

The app will be available at: **http://localhost:8000**

---

## 📁 Project Structure

```
ev-price-predictor/
├── static/                 # CSS, JS, images
├── templates/              # HTML (Jinja) templates
├── data/                   # Raw/processed datasets
├── notebooks/              # Data preprocessing & exploration
├── models/                 # Saved ML models
├── app.py                  # Flask app entry point
├── create_table.py         # SQLite3 DB schema
├── requirements.txt
└── README.md
```

---

## 🧑‍💻 Contributing

We welcome contributions! 🚀

### 📌 How to Contribute
1. Fork this repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ev-price-predictor.git
   ```
3. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Commit and push your changes:
   ```bash
   git commit -m "Added your feature"
   git push origin feature/your-feature-name
   ```
5. Submit a Pull Request.

### 📝 Contribution Guidelines
- Keep PRs focused and clean.
- Add comments and docstrings where needed.
- Run code before pushing to ensure functionality.

---


## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

📧 Resham Sai Pranathi – [pranathi9191@gmail.com](mailto:pranathi9191@gmail.com)  


---

Built with ❤️ for a greener future.


## 🙌 Contributors

Thanks to all the amazing people who help improve this project 💖

<a href="https://github.com/reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=reshamsai150/7.Analysis-and-Prediction-of-Electric-Vehicle-Costs-A-Machine-Learning-Based-Approach" />
</a>
