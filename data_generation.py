import pandas as pd
import numpy as np

def generate_customer_data(n_samples=1000):
    np.random.seed(0)
    data = pd.DataFrame({
        'Age': np.random.randint(18, 70, size=n_samples),
        'Income': np.random.normal(50000, 15000, size=n_samples),
        'SpendingScore': np.random.uniform(1, 100, size=n_samples)
    })
    return data

if __name__ == "__main__":
    customer_data = generate_customer_data()
    customer_data.to_csv("synthetic_customer_data.csv", index=False)
    