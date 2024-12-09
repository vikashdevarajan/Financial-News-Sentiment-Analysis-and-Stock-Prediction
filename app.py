import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the small English model from spacy
nlp = spacy.load("en_core_web_sm")

def fetch_stock_news(symbol, company_name):
    url = f'https://finviz.com/quote.ashx?t={symbol}&p=d'
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            news_table = soup.find('table', class_='fullview-news-outer')
            if not news_table:
                print("News table not found.")
                return pd.DataFrame()
            news_rows = news_table.findAll('tr')

            data = []
            for row in news_rows:
                date_data = row.td.text.strip().split(' ')
                if len(date_data) == 2:
                    date, time = date_data
                else:
                    date = date_data[0]
                    time = None
                headline = row.a.text.strip()

                # Perform POS tagging
                tokens = nltk.word_tokenize(headline)
                pos_tags = nltk.pos_tag(tokens)

                # Perform Named Entity Recognition (NER) using spacy
                doc = nlp(headline)
                named_entities = [(ent.text, ent.label_) for ent in doc.ents]

                # Analyze sentiment using VADER
                analyzer = SentimentIntensityAnalyzer()
                sentiment_score = analyzer.polarity_scores(headline)
                compound_score = sentiment_score['compound']

                # Categorize sentiment based on the compound score
                if compound_score >= 0.05:
                    sentiment = "positive"
                elif compound_score <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                # Append data
                data.append([date, headline, company_name, pos_tags, named_entities, sentiment])

            # Create a DataFrame from the data
            df = pd.DataFrame(data, columns=['Date', 'Headline', 'Company', 'POS_Tags', 'Named_Entities', 'Sentiment'])

            # Convert 'Date' to a proper format if necessary
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date  # Store only the date

            # Display the 5 most recent headlines with POS tags and named entities
            print("\nRecent 5 Headlines for", company_name)
            recent_headlines = df.head(5)  # Display the first 5 rows
            for index, row in recent_headlines.iterrows():
                print(f"\nHeadline {index + 1}: {row['Headline']}")
                print(f"POS Tags: {row['POS_Tags']}")
                print(f"Named Entities: {row['Named_Entities']}")

            return df
        except requests.HTTPError as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
    print(f"Failed after 3 attempts: {e}")
    return pd.DataFrame()

def append_to_excel(df, file_name='knowledgebase2.xlsx'):
    try:
        # Read existing data if the file exists
        try:
            existing_data = pd.read_excel(file_name)
            df = pd.concat([existing_data, df], ignore_index=True)
        except FileNotFoundError:
            print(f"{file_name} not found. Creating a new file.")

        # Write the updated DataFrame to the Excel file
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, index=False)
        print(f"Data successfully appended to {file_name}.")
    except Exception as e:
        print(f"Error while saving to Excel: {e}")

def main():
    company_dict = {
        1: ('TSLA', 'Tesla'),
        2: ('AAPL', 'Apple'),
        3: ('AMZN', 'Amazon'),
        4: ('MSFT', 'Microsoft'),
        5: ('GOOGL', 'Google')
    }
    print("Choose a company to analyze:")
    for key, (symbol, name) in company_dict.items():
        print(f"{key}. {name}")
    
    choice = int(input("Enter the number of the company: "))
    if choice in company_dict:
        symbol, company_name = company_dict[choice]
        print(f"Fetching data for {company_name} from {symbol}...")
        df = fetch_stock_news(symbol, company_name)
        if not df.empty:
            append_to_excel(df)
        else:
            print("No data to save. Please try again later.")
    else:
        print("Invalid choice. Please select a valid company number.")

if __name__ == "__main__":
    main()
