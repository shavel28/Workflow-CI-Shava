import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Mengaktifkan autolog lokal
mlflow.autolog()

# 2. Load data bersih (nama dipertahankan)
df = pd.read_csv("customer_preprocessed.csv")
X = df.drop("Target", axis=1)
y = df["Target"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Proses Training Otomatis
with mlflow.start_run(run_name="CI_Automated_Training") as run:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Mencatat RUN_ID ke file teks agar dibaca otomatis oleh GitHub Actions
    with open("latest_run.txt", "w") as f:
        f.write(run.info.run_id)

print("Training Otomatis Selesai!")
