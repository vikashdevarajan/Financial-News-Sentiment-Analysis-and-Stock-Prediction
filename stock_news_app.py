import requests
from bs4 import BeautifulSoup
import streamlit as st

# Function to fetch news articles based on stock ticker
def fetch_news(stock_ticker):
    # URL for fetching news from Google News
    url = f"https://www.google.com/search?q={stock_ticker}+stock+news&hl=en"
    
    # Send a request to the URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    # Check for successful request
    if response.status_code != 200:
        st.error("Failed to retrieve news articles.")
        return []
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find articles; adjust the selectors based on the structure of the website you're scraping
    articles = soup.find_all('div', class_='BVG0Nb')  # This class may vary; inspect the page to find the right one
    
    news_list = []
    for article in articles[:5]:  # Get top 5 articles
        title = article.find('h3').text if article.find('h3') else 'No Title'
        link = article.find('a')['href'] if article.find('a') else '#'
        news_list.append((title, link))
    
    return news_list

# Streamlit application layout
st.title("Stock News Fetcher")
st.write("Enter a stock ticker symbol to fetch recent news articles:")

# User input for stock ticker
stock_ticker = st.text_input("Stock Ticker (e.g., AAPL for Apple):")

if st.button("Fetch News"):
    if stock_ticker:
        # Fetch and display news articles
        news_articles = fetch_news(stock_ticker)
        if news_articles:
            st.write(f"Top 5 recent news articles about {stock_ticker}:")
            for title, link in news_articles:
                st.write(f"- [{title}]({link})")
        else:
            st.warning("No news found.")
    else:
        st.warning("Please enter a stock ticker.")