## Subscriber Churn Prediction – Minor Project

This project implements an **Artificial Neural Network (ANN)** to predict subscriber churn and exposes it through a simple **Flask web application**.

The model is trained on the classic `Churn_Modelling.csv` dataset (bank customers) and can be extended to other service domains such as telecom, hotels, and subscription services.

### 1. Project structure

- `train_model.py` – trains the ANN and saves the model and scaler.
- `app.py` – Flask web app that loads the trained model and provides a web form for predictions.
- `templates/index.html` – input form for customer details.
- `templates/result.html` – displays prediction and churn probability.
- `requirements.txt` – Python dependencies.
- `data/Churn_Modelling.csv` – dataset file (you must place it here).
- `models/` – saved ANN model and scaler (created after training).

### 2. Setup (Windows)

1. **Open PowerShell** and go to your project folder:

   ```powershell
   cd C:\sowmya_minor_Project
   ```

2. (Recommended) **Create a virtual environment**:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

4. **Add the dataset**:

   - Create a `data` folder inside the project if it does not exist.
   - Copy `Churn_Modelling.csv` into `C:\sowmya_minor_Project\data`.

### 3. Train the ANN model

Run the training script once to train and save the model and scaler:

```powershell
cd C:\sowmya_minor_Project
python train_model.py
```

This will:

- Train the ANN using the pre-processing pipeline (encoding + scaling).
- Print the **confusion matrix** and **accuracy**.
- Save the trained model to `models/bank_churn_ann.keras`.
- Save the scaler to `models/scaler.pkl`.

### 4. Run the web application

After training completes successfully:

```powershell
cd C:\sowmya_minor_Project
python app.py
```

By default the app runs on port 5000. Open a browser and go to:

`http://localhost:5000`

You will see a form where you can enter:

- Credit score, geography, gender
- Age, tenure, balance
- Number of products, has credit card, is active member
- Estimated salary

Submitting the form shows:

- **Churn probability (percentage)**
- Whether the customer is at **high risk** or **low risk** of churn
- A short qualitative explanation suitable for your project report

### 5. Connecting to your abstract / report

- **Data processing** – implemented in `train_model.py`:
  - Cleaning by removing identifier columns.
  - Handling categorical variables (`Geography`, `Gender`) via fixed mappings.
  - Feature scaling using `StandardScaler`.
- **ANN model** – implemented in `train_model.py` (sequential network with two hidden layers and sigmoid output).
- **Evaluation** – confusion matrix and accuracy are printed during training.
- **Deployment / website** – implemented in `app.py` with templates in `templates/`.

You can include screenshots of:

- The home page (`index.html`) form.
- Example prediction result page (`result.html`).
- Terminal output showing confusion matrix and accuracy.

