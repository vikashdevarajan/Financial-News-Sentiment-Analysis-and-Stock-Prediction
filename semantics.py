import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob


nlp = spacy.load("en_core_web_sm")


knowledge_base = pd.read_excel('knowledgebase2.xlsx')
print("Loaded knowledge base from knowledgebase2.xlsx.")


predefined_companies = ['Apple', 'Tesla', 'Amazon', 'Microsoft', 'Google', 'Elon Musk']



common_spelling_corrections = {
    "teh": "the",
    "negotive": "negative",
    "positivee": "positive",
    "splling": "spelling"
}


def correct_spelling(query):
    query_lower = query.lower()
    company_name_in_query = None
    for company in predefined_companies:
        if company.lower() in query_lower:
            company_name_in_query = company
            query_lower = re.sub(r'\b' + re.escape(company.lower()) + r'\b', "COMPANY_PLACEHOLDER", query_lower)

    corrected_words = []
    for word in query_lower.split():
        if word in common_spelling_corrections:
            corrected_words.append(common_spelling_corrections[word])
        elif word != "company_placeholder":
            corrected_words.append(str(TextBlob(word).correct()))
        else:
            corrected_words.append(word)

    corrected_query = ' '.join(corrected_words)
    if company_name_in_query:
        corrected_query = corrected_query.replace("COMPANY_PLACEHOLDER", company_name_in_query)

    original_words = set(query.lower().split())
    corrected_words_set = set(corrected_query.lower().split())
    uncorrected_words = original_words - corrected_words_set
    if uncorrected_words:
        print(f"Warning: The following words were not corrected: {', '.join(uncorrected_words)}")

    return corrected_query


def understand_query(query):
    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    chunks = [chunk.text for chunk in doc.noun_chunks]
    print(f"Entities identified: {entities}")
    print(f"Keywords identified: {keywords}")
    print(f"Noun chunks identified: {chunks}")
    return entities, keywords, chunks


def retrieve_relevant_headlines(query, knowledge_base, top_n=5):
    company, sentiment = extract_company_and_sentiment(query)

    if company is None:
        print("No company found in the query. Please specify a company.")
        return

   
    company_headlines = knowledge_base[knowledge_base['Company'].str.lower() == company.lower()]

    
    if sentiment:
        company_headlines = company_headlines[company_headlines['Sentiment'].str.lower() == sentiment.lower()]

   
    company_headlines = company_headlines.drop_duplicates(subset=['Headline'])

    
    cleaned_query = clean_text(query)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [cleaned_query] + [clean_text(str(headline)) for headline in company_headlines['Headline']]
    )
    query_vector = tfidf_matrix[0]
    headline_vectors = tfidf_matrix[1:]

   
    similarity_scores = cosine_similarity(query_vector, headline_vectors).flatten()
    company_headlines['Similarity Score'] = similarity_scores

    
    company_headlines_sorted = company_headlines.sort_values(
        by='Similarity Score', ascending=False
    ).head(top_n)

  
    print(f"\nTop {top_n} relevant headlines based on your query for '{company}':\n")
    for index, row in company_headlines_sorted.iterrows():
        
        print(f"Headline: {row['Headline']}")
        print(f"Company: {row['Company']}")
        print(f"Sentiment: {row['Sentiment']}")
        print(f"Similarity Score: {row['Similarity Score']:.4f}\n")


def extract_company_and_sentiment(query):
    query = query.lower()
    company = None
    sentiment = None

    if 'positive' in query:
        sentiment = 'positive'
    elif 'negative' in query:
        sentiment = 'negative'

    for known_company in predefined_companies:
        if known_company.lower() in query:
            company = known_company
            break

    return company, sentiment


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text


def main():
    query = input("Enter your query: ")

    corrected_query = correct_spelling(query)
    print(f"Corrected query: {corrected_query}")

    entities, keywords, chunks = understand_query(corrected_query)
    print(f"Noun chunks used in analysis: {chunks}")

    retrieve_relevant_headlines(corrected_query, knowledge_base)

if __name__ == "__main__":
    main()
