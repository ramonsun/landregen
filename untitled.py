import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Create a simulated dataset
np.random.seed(0)# For reproducibility
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)

# Display the first few rows of the dataframe
df.head()
from sklearn.model_selection import train_test_split

X = df[['feature1', 'feature2']]  # Features
y = df['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


