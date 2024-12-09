import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")


train, test = train_test_split(df, test_size=0.2, random_state=42)


data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)


list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    if isinstance(text, float):
        text = ''
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem words
    return ' '.join(words)


for index in new_Index:
    data[index] = data[index].astype(str).apply(preprocess_text)


headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))


tfidfvector = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3)
traindataset = tfidfvector.fit_transform(headlines)


scaler = StandardScaler(with_mean=False)
traindataset_scaled = scaler.fit_transform(traindataset)


test_data = test.iloc[:, 2:27]
test_data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
test_data.columns = new_Index

for index in new_Index:
    test_data[index] = test_data[index].astype(str).apply(preprocess_text)


test_transform = []
for row in range(0, len(test_data.index)):
    test_transform.append(' '.join(str(x) for x in test_data.iloc[row, 0:25]))


test_dataset = tfidfvector.transform(test_transform)
test_dataset_scaled = scaler.transform(test_dataset)


random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=4,
    min_samples_leaf=1,
    criterion='entropy',
    class_weight='balanced_subsample',
    random_state=42
)

logistic_regression = LogisticRegression(
    max_iter=300,
    class_weight='balanced',
    random_state=42
)

# Train the models
random_forest.fit(traindataset_scaled, train['Label'])
logistic_regression.fit(traindataset_scaled, train['Label'])


print("Select a company from the list:")
print("1. Amazon")
print("2. Tesla")
print("3. Apple")
print("4. Microsoft")
print("5. Google")
choice = int(input("Enter the number corresponding to your choice: "))


company_dict = {
    1: "Amazon",
    2: "Tesla",
    3: "Apple",
    4: "Microsoft",
    5: "Google"
}
selected_company = company_dict.get(choice, None)

if not selected_company:
    print("Invalid choice. Please restart and choose a valid number.")
    exit()


knowledgebase = pd.read_excel('knowledgebase2.xlsx')

# Filter headlines for the selected company and ensure only up to 25 are used
company_headlines = knowledgebase[knowledgebase['Company'] == selected_company]['Headline'].tolist()[:25]

# Preprocess the headlines
preprocessed_headlines = [preprocess_text(headline) for headline in company_headlines]

# Transform the headlines into TF-IDF vectors
company_tfidf = tfidfvector.transform(preprocessed_headlines)
company_tfidf_scaled = scaler.transform(company_tfidf)

# Make predictions using both models and average the probabilities
rf_predictions = random_forest.predict_proba(company_tfidf_scaled)[:, 1]
lr_predictions = logistic_regression.predict_proba(company_tfidf_scaled)[:, 1]
final_predictions = (rf_predictions + lr_predictions) / 2 > 0.5
final_predictions = final_predictions.astype(int)


if final_predictions.mean() > 0.5:
    print(f"The prediction indicates that {selected_company}'s stock is likely to go up.")
else:
    print(f"The prediction indicates that {selected_company}'s stock is likely to go down.")
