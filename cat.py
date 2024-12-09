import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import requests
from bs4 import BeautifulSoup
import networkx as nx
import pandas as pd


original_url = "https://greatergood.berkeley.edu/article/item/our_best_education_articles_of_2021"


table_url = "https://www.educationworld.in/ew-india-school-rankings-2023-24-top-best-schools-in-india/"


response = requests.get(original_url)
soup = BeautifulSoup(response.content, 'html.parser')


text_corpus = soup.get_text()


def extract_table_from_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    if table:
        df = pd.read_html(str(table))[0]  
        return df
    return None


def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

create_word_cloud(text_corpus)


def create_wordnet_diagram(words):
    G = nx.Graph()
    for word in words:
        synsets = wn.synsets(word)
        for synset in synsets:
            G.add_node(synset.name(), color='lightblue')
            for lemma in synset.lemmas():
                G.add_node(lemma.name(), color='lightgreen')
                G.add_edge(synset.name(), lemma.name(), color='gray')

    
    node_colors = [G.nodes[node]['color'] for node in G]
    node_labels = {node: node.split('.')[0] for node in G}

   
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    plt.figure(figsize=(15, 18))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, node_color=node_colors, font_size=8, edge_color='gray', linewidths=0.5, font_color='black')
    plt.title('WordNet Diagram')
    plt.show()


words = input("Enter three words separated by spaces: ").split()
create_wordnet_diagram(words)


def count_word_occurrences(df, word):
    if 'State' in df.columns:
        state_column = df['State'].astype(str)
        word_count = state_column.str.contains(word, case=False, na=False).sum()
        return word_count
    else:
        return None

df = extract_table_from_webpage(table_url)
if df is not None:
    word_to_check = input("Enter a word to count its occurrences in the 'State' column: ")
    word_count = count_word_occurrences(df, word_to_check)
    if word_count is not None:
        print(f"Occurrences of the word '{word_to_check}' in the 'State' column: {word_count}")
    else:
        print("The 'State' column is not present in the table.")
else:
    print("No table found on the webpage.")


def kleene_closure_patterns(text):
    tokens = nltk.word_tokenize(text)
    
    
    words_with_star = [word for word in tokens if re.fullmatch(r'\b\w*a\w*\b', word)]
    
   
    words_with_plus = [word for word in tokens if re.fullmatch(r'\b\w*a+\w*\b', word)]
    
    print("15 Words with Kleene * Pattern (zero or more 'a's):")
    print(words_with_star[:15])
    
    print("\n15 Words with Kleene + Pattern (one or more 'a's):")
    print(words_with_plus[:15])

kleene_closure_patterns(text_corpus)


def apply_regex_patterns(text):
    patterns = {
        '[a-zA-Z]+': re.findall(r'[a-zA-Z]+', text),
        '[A-Z][a-z]*': re.findall(r'[A-Z][a-z]*', text),
        'p[aeiou]{,2}t': re.findall(r'p[aeiou]{,2}t', text),
        '\d+(\.\d+)?': re.findall(r'\d+(\.\d+)?', text),
        '([^aeiou][aeiou][^aeiou])*': re.findall(r'([^aeiou][aeiou][^aeiou])*', text),
        '\w+|[^\w\s]+': re.findall(r'\w+|[^\w\s]+', text)
    }
    
    for pattern, matches in patterns.items():
        print(f"Pattern {pattern}: {matches[:15]}")  

apply_regex_patterns(text_corpus)
