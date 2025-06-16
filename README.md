# AI-Based Energy Usage Prediction System

This capstone project presents an AI-powered solution to predict household energy consumption using LSTM (Long Short-Term Memory) neural networks. The system is designed to enhance energy efficiency and provide actionable insights for smart energy management across different zones of a household.

## 🚀 Live App
👉 [Click here to try the app](https://pradyumn.streamlit.app/)

## 🔍 Project Overview

The system utilizes historical power consumption data to forecast energy usage in three primary areas:
- **AC/Heater Zone**
- **Kitchen Appliances**
- **Laundry Appliances**

Predictions are made using separate LSTM models trained on the `Individual Household Electric Power Consumption` dataset. The application features a user-friendly interface built using Streamlit, allowing for intuitive interaction and visualization.

## 🧠 Key Features

- LSTM-based deep learning models for time series prediction
- Zone-wise prediction for targeted energy management
- Real-time visualization of predicted vs. actual consumption
- Easy-to-use web interface using Streamlit

## 🗂️ Project Structure

```
capstone_project/
│
├── datset/                    # dataset of the project
│   ├── household_power_consumption.txt
│
├── app.py                     # Streamlit application
├── lstmmodel.ipynb            # Notebook for model training
├── model_ac_heater.h5         # Trained model for AC/Heater
├── model_kitchen.h5           # Trained model for Kitchen
├── model_laundry.h5           # Trained model for Laundry
└── requirements.txt           # Python dependencies
```

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

## 🧪 Model Training (Optional)

If you wish to retrain the models:

- Open `lstmmodel.ipynb` in Jupyter Notebook or VS Code.
- Follow the notebook instructions to train new LSTM models.
- Save the models with appropriate names (e.g., `model_kitchen.h5`).

## 📊 Dataset Used

The models are trained using the publicly available [Individual Household Electric Power Consumption Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) from Kaggle.

## 🚀 Future Enhancements

- Integration with IoT sensors for real-time data collection
- Extension to commercial building energy systems
- Deployment to cloud platforms (AWS/GCP) for scalability

## 🧑‍💻 Author

**Devanshu Sawarkar**  
**Pratham Agrawal**     
**Devansh Motghare**    
**Pradyumn Waghmare**   
**Ayush Gupte** 

## 📄 License

This project is licensed under the MIT License.
