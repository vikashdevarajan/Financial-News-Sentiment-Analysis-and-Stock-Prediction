# main.py
import streamlit as st
import pandas as pd
from app import fetch_stock_news, append_to_excel
from semantics import correct_spelling, understand_query, retrieve_relevant_headlines
from prediction import preprocess_text, tfidfvector, scaler, random_forest, logistic_regression

# Load the knowledge base (Excel file)
knowledge_base_file = 'knowledgebase2.xlsx'
knowledge_base = pd.read_excel(knowledge_base_file)

# Dictionary for selecting companies
company_dict = {
    1: ('TSLA', 'Tesla'),
    2: ('AAPL', 'Apple'),
    3: ('AMZN', 'Amazon'),
    4: ('MSFT', 'Microsoft'),
    5: ('GOOGL', 'Google')
}

st.title("Stock Analysis & Prediction")
st.sidebar.header("Select a Company")

# Company selection
company_choice = st.sidebar.selectbox(
    "Choose a company to analyze:",
    list(company_dict.keys()),
    format_func=lambda x: company_dict[x][1]
)

# Retrieve symbol and company name
symbol, company_name = company_dict[company_choice]

st.write(f"**Selected Company:** {company_name}")

# Fetch and display latest headlines
if st.button("Fetch Latest Headlines"):
    st.write(f"Fetching latest news for {company_name}...")
    df = fetch_stock_news(symbol, company_name)
    if df is not None and not df.empty:
        append_to_excel(df, file_name=knowledge_base_file)
        st.write("Headlines successfully fetched and saved to the knowledge base.")
        st.dataframe(df.head())
    else:
        st.write("No new headlines retrieved. Please try again later.")

# Query input for semantic analysis
st.header("Query Analysis")
query = st.text_input("Enter your query about the stock or company:")

if st.button("Analyze Query"):
    if query:
        st.write(f"Analyzing the query: {query}")
        corrected_query = correct_spelling(query)
        st.write(f"**Corrected Query:** {corrected_query}")

        entities, keywords, chunks = understand_query(corrected_query)
        st.write(f"**Identified Entities:** {entities}")
        st.write(f"**Extracted Keywords:** {keywords}")
        st.write(f"**Identified Noun Chunks:** {chunks}")

        st.write("Retrieving relevant headlines...")
        try:
            # Display relevant headlines based on the query
            top_n = 5  # Number of top relevant headlines to display
            relevant_headlines = retrieve_relevant_headlines(corrected_query, knowledge_base, top_n)
            if relevant_headlines is not None and not relevant_headlines.empty:
                st.table(relevant_headlines[['Date', 'Time', 'Headline', 'Sentiment', 'Similarity Score']])
            else:
                st.write("No relevant headlines found based on the query.")
        except Exception as e:
            st.write(f"Error retrieving headlines: {e}")

# Prediction section
st.header("Stock Movement Prediction")

if st.button("Predict Stock Movement"):
    st.write(f"Predicting stock movement for {company_name}...")

    # Filter headlines for the selected company from the knowledge base
    company_headlines = knowledge_base[knowledge_base['Company'] == company_name]['Headline'].tolist()

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

    # Display the prediction result
    if final_predictions.mean() > 0.5:
        st.write(f"The prediction indicates that **{company_name}'s stock is likely to go up.**")
    else:
        st.write(f"The prediction indicates that **{company_name}'s stock is likely to go down.**")
