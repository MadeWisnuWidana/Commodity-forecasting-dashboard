# 📊 Commodity Price Prediction Dashboard

This is an interactive web dashboard built with **Streamlit** to forecast the prices of staple commodities in **Indonesia**. The application leverages multiple machine learning and statistical models to provide dynamic, insightful predictions.

> ⚠️ **Note**: The dataset `Harga_Nasional_Pivot.csv` is **not included** in this repository. You must provide your own data file with the specified format (see [📄 Data File Information](#-data-file-information)).

---

## ✨ Key Features

- **Interactive Dashboard**  
  A user-friendly interface built with Streamlit for seamless navigation and interaction.

- **Multiple Forecasting Models**  
  Choose from three powerful forecasting models:
  - **Prophet**: Robust time-series forecasting developed by Facebook, ideal for seasonality and holiday effects.
  - **XGBoost**: A powerful gradient-boosting algorithm adapted for time-series prediction.
  - **SARIMA**: A classical statistical model (Seasonal ARIMA) effective for seasonal time-series data.

- **Dynamic User Input**  
  Users can select:
  - The desired **commodity**
  - The **forecasting model**
  - The **forecast duration**

- **Rich Visualizations**  
  Forecast results and model components are displayed using **interactive Plotly charts**.

- **Performance Metrics**  
  Each model is evaluated using:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)

- **Holiday Effects**  
  Indonesian national holidays are incorporated to improve forecast accuracy.

---

## 🛠️ Technologies Used

| Purpose               | Tools / Libraries                         |
|-----------------------|-------------------------------------------|
| Web Framework         | Streamlit                                 |
| Data Manipulation     | Pandas, NumPy                             |
| Forecasting Models    | Prophet, XGBoost, statsmodels (SARIMA)    |
| Data Visualization    | Plotly                                    |
| Evaluation Metrics    | scikit-learn                              |

---

## 🚀 How to Run the Application

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
## 🚀 How to Set Up and Run the Application
```
### 2️⃣ Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
### 3️⃣ Install Dependencies
Create a requirements.txt file and include the following content:
```bash
streamlit
pandas
numpy
plotly
prophet
statsmodels
xgboost
scikit-learn
```
Then install all dependencies with:
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Data File
Make sure to place the Harga_Nasional_Pivot.csv file in the root directory of the project.
  ### ℹ️ See the 📄 Data File Information section below for the required format.

### 5️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

### 📄 Data File Information
Your data file must meet the following requirements:
File Name: Harga_Nasional_Pivot.csv
Location: Must be placed in the root directory of the project
Format:
- CSV format
- Must contain a column named Tanggal in dd/mm/yyyy format
- Other columns should represent commodity names (e.g., Beras, Gula Pasir, Minyak Goreng) with numeric price values

✅ Example Structure:
```
Tanggal,Beras,Gula Pasir,Minyak Goreng
20/01/2024,15000,18000,22000
21/01/2024,15100,18050,21900
22/01/2024,15200,18100,22100
```

📬 Contact
For questions, collaboration, or feedback, feel free to reach out:
📧 madewisnuwidana59@gmail.com

![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)
![Forecasting Models: Prophet | XGBoost | SARIMA](https://img.shields.io/badge/Models-Prophet%20%7C%20XGBoost%20%7C%20SARIMA-blue)


