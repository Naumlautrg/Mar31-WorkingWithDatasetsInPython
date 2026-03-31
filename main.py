import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo

# 1. SETUP & DATA LOADING
#url = "https://github.com/setnormTJC/MIS-classification-algorithm-project/raw/master/Loan_approval_data.csv" #note the RAW!
#data = pd.read_csv(url)
adult = fetch_ucirepo(id=2)

factorsToTrack = [
    'age', 'workclass', 'education', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'native-country'
]

#X = data[factorsToTrack]
#y = data['income'].map({'>50K': 0, '<=50K': 1})
X = adult.data.features
y = adult.data.targets['income'].str.strip('.').map({'<=50K': 0, '>50K': 1})

categorical_options = {}
categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

for col in categorical_cols:
  categorical_options[col] = sorted(X[col].dropna().unique())

X = pd.get_dummies(X)

# 2. THE TRAIN-TEST SPLIT
# 80% to train the "Robot", 20% to test it on people it hasn't met.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAINING
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 5. EVALUATION
train_acc = model.score(X_train, y_train) * 100
test_acc = model.score(X_test, y_test) * 100

print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy (Unseen Data): {test_acc:.2f}%")

# 6. VISUALIZATION (Weights & Confusion Matrix)
weights = model.coef_[0]
feature_names = X.columns

# We have to group the categories back together into a single column
importance_df = pd.DataFrame({
    'feature': feature_names,
    'weight': weights
})

# Map the features back
def get_base_feature(name):
  # If the feature name starts with the name of ts base, return the base
  # Otherwise return the name (for numeric columns, in this case "age")
  for base in factorsToTrack:
    if name.startswith(base):
      return base
  return name

importance_df['base_feature'] = importance_df['feature'].apply(get_base_feature)

# Get the Aggregate (sum absolute importance per category)
grouped = importance_df.groupby('base_feature')['weight'].apply(lambda x: x.abs().sum())

# Sort grouped data
grouped = grouped.sort_values(ascending=True)

# Drop untracked fields
grouped = grouped.drop('capital-gain')
grouped = grouped.drop('capital-loss')
grouped = grouped.drop('fnlwgt')

#Plot 1: Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(grouped.index, grouped.values, color='darkblue')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Weights of Income Bracket Influences")
plt.tight_layout()
plt.savefig("Weights.png")

# Plot 1: Feature Importance
#plt.figure(figsize=(10, 5))
#feature_names = X.columns
#plt.barh(feature_names, weights, color='darkblue')
#plt.axvline(0, color='black', linewidth=0.8)
#plt.title("What Influences one's Income Bracket?")
#plt.tight_layout()
#plt.savefig("Weights.png")

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50k'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Results)")
plt.savefig("Confusion_Matrix.png")
plt.show()

# 7. INTERACTIVE PREDICTION
def validate_input(prompt, valid_options):
  while True:
    user_input = input(prompt).strip()

    for option in valid_options:
      if user_input.lower() == option.lower():
        return option

    print("Invalid input. Choose from:")
    print(", ".join(valid_options))

def validate_age():
  while True:
    try:
      age = float(input("Age: "))
      return age
    except ValueError:
      print("Please enter a valid number.")

def predict_incomeBracket():
    print("\n--- Income Analysis ---")

    age = validate_age()
    workclass = validate_input("Workclass: ", categorical_options['workclass'])
    edu = validate_input("Education: ", categorical_options['education'])
    marital = validate_input("Marital Status: ", categorical_options['marital-status'])
    occupation = validate_input("Occupation: ", categorical_options['occupation'])
    relationship = validate_input("Relationship: ", categorical_options['relationship'])
    race = validate_input("Race: ", categorical_options['race'])
    sex = validate_input("Sex: ", categorical_options['sex'])
    native_country = validate_input("Native Country: ", categorical_options['native-country'])

    user_dict = {
        'age': age,
        f'workclass_{workclass}': 1,
        f'education_{edu}': 1,
        f'marital-status_{marital}': 1,
        f'occupation_{occupation}': 1,
        f'relationship_{relationship}': 1,
        f'race_{race}': 1,
        f'sex_{sex}': 1,
        f'native-country_{native_country}': 1
    }

    # Convert dict to DataFrame
    user_df = pd.DataFrame([user_dict])

    # Fill in missing columns
    for col in feature_columns:
      if col not in user_df:
        user_df[col] = 0

    user_df = user_df[feature_columns]

    pred = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][pred] * 100

    verdict = ">50K" if pred == 1 else "<=50K"
    print(f"\nDecision: {verdict} ({prob:.1f}% confidence)")

predict_incomeBracket()
