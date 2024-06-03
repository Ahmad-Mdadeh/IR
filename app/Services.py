from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import numpy as np
import joblib
import json
import os
import json
from nltk.tokenize import word_tokenize
from dateutil.parser import ParserError
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import words
from dateutil import parser
from tqdm import tqdm
import pandas as pd
import string
import json
import nltk
import os
import re

# ------------------------------Division of tokens----------------------------------
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

    # ex:
    # text = "This is a sample text for tokenization."
    # tokens = tokenize_text(text)
    # print(tokens)
    # solution:
    # ['This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# ----------------------------------------------------------------------------------

# ---------------Remove punctuation and arrows from the list tokens------------------
def remove_punctuation(tokens):
    additional_punctuation = "’→←↔"
    all_punctuation = string.punctuation + additional_punctuation
    punctuation_regex = re.compile(f"[{re.escape(all_punctuation)}]")

    def should_skip(token):
        if re.match(r'^\w+/\w+$', token):
            return True
        if re.match(r'^\d+/\d+$', token):
            return True
        return False

    cleaned_tokens = [
        token if should_skip(token) else punctuation_regex.sub('', token)
        for token in tokens
    ]
    cleaned_tokens = [token for token in cleaned_tokens if token]

    return cleaned_tokens

    # ex:
    # tokens = ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'with', 'punctuation', '.', '5/5', 'is', 'an', 'example', 'of', 'a', 'fraction', '.']
    # cleaned_tokens = remove_punctuation(tokens)
    # print(cleaned_tokens)
    # solution:
    # ['Hello', 'world', 'This', 'is', 'a', 'sample', 'text', 'with', 'punctuation', '5/5', 'is', 'an', 'example', 'of', 'a', 'fraction']
# ----------------------------------------------------------------------------------

# -----------------------------Remove url from tokens-------------------------------
def remove_urls(tokens):
    url_regex = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )    
    cleaned_tokens = [token for token in tokens if not url_regex.match(token)]
    
    return cleaned_tokens
    # ex:
    # tokens = ['Hello', 'world', 'visit', 'my', 'website', 'at', 'https://example.com', 'for', 'more', 'information', '!', 'You', 'can', 'also', 'email', 'me', 'at', 'john@example.com']
    # cleaned_tokens = remove_urls(tokens)
    # print(cleaned_tokens)
    # solution:
    # ['
    # ', 'world', 'visit', 'my', 'website', 'at', 'for', 'more', 'information', '!', 'You', 'can', 'also', 'email', 'me', 'at']
# ----------------------------------------------------------------------------------

# ------------------------------convert_to_lowercase--------------------------------
def convert_to_lowercase(tokens):
    lowercase_tokens = [token.lower() for token in tokens]
    return lowercase_tokens

    #ex:
    # tokens = ["Hello", "WORLD", "PyThOn", "ProgRaMMing"]
    # lowercase_tokens = convert_to_lowercase(tokens)
    # print(lowercase_tokens)
    # solution:
    # ['hello', 'world', 'python', 'programming']
# ----------------------------------------------------------------------------------
# ---------------------------------
def process_and_standardize_dates_brackets(tokens):
    processed_tokens = []
    date_pattern = re.compile(
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(\d{4}[-/]\d{1,2}[-/]\d{1,2})|'
        r'(\d{1,2}\s+\w+\s+\d{2,4})|(\w+\s+\d{1,2},\s+\d{2,4})'
    )

    for token in tokens:
        token = re.sub(r'[()]', '', token)
        if re.match(r'^\d+$', token):
            processed_tokens.append(token)
        else:
            if date_pattern.match(token):
                try:
                    parsed_date = parser.parse(token, fuzzy=True)
                    standardized_date = parsed_date.strftime('%Y-%m-%d')
                    processed_tokens.append(standardized_date)
                except (ParserError, ValueError, OverflowError):
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
    return processed_tokens

    #ex:
    # tokens = [
    # "(1234)", "(12-05-2021)", "Hello", "12/31/1999", "2021-12-05", "(01 Jan 2022)", 
    # "(March 3, 2020)", "12345", "2022/03/05", "not-a-date"
    # ]
    # processed_tokens = process_and_standardize_dates_brackets(tokens)
    # print(processed_tokens)
    # solution:
    # ['1234', '2021-12-05', 'Hello', '1999-12-31', '2021-12-05', '2022-01-01', '2020-03-03', '12345', '2022-03-05', 'not-a-date']
# --------------------------------------------------------------------------------------------------
# ---------------------------------------get_wordnet_pos--------------------------------------------
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)
    # ex:
    # words = ["running", "quickly", "beautiful", "car"]
    # for word in words:
    #     pos = get_wordnet_pos(word)
    #     print(f"Word: {word}, WordNet POS: {pos}")
    # solution:
    # Word: running, WordNet POS: v
    # Word: quickly, WordNet POS: r
    # Word: beautiful, WordNet POS: a
    # Word: car, WordNet POS: n

# --------------------------------------------------------------------------------------------------
# ------------------------------------remove_stopwords----------------------------------------------
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens
    # ex:
    # tokens = ["This", "is", "a", "simple", "example", "showing", "how", "to", "remove", "stopwords"]
    # filtered_tokens = remove_stopwords(tokens)
    # print(filtered_tokens)
    # solution:
    # ['simple', 'example', 'showing', 'remove', 'stopwords']
# --------------------------------------------------------------------------------------------------
# -------------------------------remove_phonetic_notation--------------------------------------------
def remove_phonetic_notation(text):
    cleaned_text = ""
    in_phonetic_notation = False 
    
    for char in text:
        if char == '/':
            in_phonetic_notation = not in_phonetic_notation
        elif not in_phonetic_notation:
            cleaned_text += char
    return cleaned_text
    # ex:
    # text_with_phonetic_notation = "The word /pɜːraɪˈɒdɪk/ is often used in scientific contexts."
    # cleaned_text = remove_phonetic_notation(text_with_phonetic_notation)
    # print(cleaned_text)
    # solution:
    # The word  is often used in scientific contexts.
# --------------------------------------------------------------------------------------------------
# ----------------------------------------stem_tokens-----------------------------------------------
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
    # ex:
    # tokens = ["running", "jumps", "easily", "fairly"]
    # stemmed_tokens = stem_tokens(tokens)
    # print(stemmed_tokens)
    # solution:
    # ['run', 'jump', 'easili', 'fairli']
# --------------------------------------------------------------------------------------------------
# -------------------------------------lemmatize_tokens----------------------------------------------
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return lemmatized_tokens
    # ex:
    # tokens = ["running", "jumps", "easily", "fairly", "better", "cars"]
    # lemmatized_tokens = lemmatize_tokens(tokens)
    # print(lemmatized_tokens)
    # solution:
    # ['run', 'jump', 'easily', 'fairly', 'good', 'car']
# --------------------------------------------------------------------------------------------------
# ---------------------------------------join_tokens-------------------------------------------------
def join_tokens(tokens):
    joined_text = ""
    for token in tokens:
        joined_text += token + " "
    return joined_text.rstrip() 
    # ex:
    # tokens = ["This", "is", "a", "sample", "sentence"]
    # joined_text = join_tokens(tokens)
    # print(joined_text)
    # solution:
    # This is a sample sentence
# --------------------------------------------------------------------------------------------------
# ------------------------------------correct_spelling----------------------------------------------
def correct_spelling(tokens):
    spell = SpellChecker()
    corrected_tokens = [spell.correction(token) if token not in spell else token for token in tokens]
    return corrected_tokens
    # ex:
    # tokens = ["speling", "corect", "misspel", "accidently"]
    # corrected_tokens = correct_spelling(tokens)
    # print(corrected_tokens)
    # solution:
    # ['spelling', 'correct', 'misspelled', 'accidentally']
# --------------------------------------------------------------------------------------------------
# ------------------------------------remove_equations----------------------------------------------
def remove_equations(text):
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\.*?\\', '', text)
    return text
    # ex:
    # text_with_equations = "This is an example with equations $$x^2 + y^2 = r^2$$ and $E=mc^2$."
    # text_without_equations = remove_equations(text_with_equations)
    # print(text_without_equations)
    # sollution:
    # This is an example with equations  and .
# --------------------------------------------------------------------------------------------------
# ---------------------------------------process_text-----------------------------------------------
def process_text(text):
    text_without_phonetic = remove_phonetic_notation(text)
    text_without_equations = remove_equations(text_without_phonetic)
    tokens = tokenize_text(text_without_equations)
    tokens = convert_to_lowercase(tokens)
    tokens = remove_urls(tokens)
    tokens = remove_punctuation(tokens)
    tokens = process_and_standardize_dates_brackets(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    tokens = join_tokens(tokens)
    return tokens
# --------------------------------------------------------------------------------------------------
# -------------------------------------------read_data-----------------------------------------------
def read_data(file_path):
    documents = []  
    ids = []     

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  
            for line in tqdm(lines, desc="Reading lines", ncols=100, colour='blue'):
                try:
                    data = json.loads(line)  
                    documents.append(data['text'])  
                    ids.append(data['_id'])       
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line in file {file_path}: {e}")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing file {file_path}: {e}")
    return documents, ids
# --------------------------------------------------------------------------------------------------
# --------------------------------------load_inverted_index-----------------------------------------
def load_inverted_index(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            inverted_index_data = json.load(file)
            inverted_index = {term: (np.array(doc_ids), np.array(scores)) for term, (doc_ids, scores) in inverted_index_data.items()}
        return inverted_index
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the inverted index from file {file_path}: {e}")
        return None
# --------------------------------------------------------------------------------------------------
# ------------------------------------------search_query--------------------------------------------
def search_query(query, vectorizer, inverted_index):
    # vectorizer = TfidfVectorizer()
    cleaned_query = process_text(query)
    query_vec = vectorizer.transform([cleaned_query])

    scores = defaultdict(float)
    
    for idx, value in zip(query_vec.indices, query_vec.data):
        term = vectorizer.get_feature_names_out()[idx]
        if term in inverted_index:
            doc_ids, term_scores = inverted_index[term]
            for doc_id, score in zip(doc_ids, term_scores):
                scores[doc_id] += value * score

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    doc_ids = [doc_id for doc_id, _ in sorted_scores]
    return doc_ids

# # --------------------------------------------------------------------------------------------------
# # ------------------------------------extract_text_by_ids-------------------------------------------
def extract_text_by_ids(jsonl_file, id_list):
    texts = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            _id = data['_id']
            if _id in id_list:
                texts.append(data['text'])
    return texts
# # --------------------------------------------------------------------------------------------------
# # ---------------------------------------InitializationQuora---------------------------------------------
def InitializationQuora():
    documents, ids = read_data(r'C:\Users\ASUS\Desktop\ir_project\app\data_processing\Quora\corpus-processed.jsonl')
    inverted_index=r'C:\Users\ASUS\Desktop\ir_project\app\data_processing\Quora\inverted_index.json'
    loaded_compressed_index = load_inverted_index(inverted_index)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
     
    return loaded_compressed_index, vectorizer
# # --------------------------------------------------------------------------------------------------
# # ---------------------------------------InitializationLotte---------------------------------------------
def InitializationLotte():
    documents, ids = read_data(r'C:\Users\ASUS\Desktop\ir_project\data_set\Lotte\collection - Copy.jsonl')
    inverted_index=r'C:\Users\ASUS\Desktop\ir_project\app\data_processing\Lotte\inverted_index.json'
    loaded_compressed_index = load_inverted_index(inverted_index)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
     
    return loaded_compressed_index, vectorizer
# # --------------------------------------------------------------------------------------------------
# # ----------------------------------------------main------------------------------------------------
def main(query,type_data_set):
    if type_data_set == "Quora":
        Document = r'C:\Users\ASUS\Desktop\ir_project\data_set\Quora\corpus.jsonl'
        loaded_compressed_index, vectorizer = InitializationQuora()
    elif type_data_set == "Lotte":
        Document = r'C:\Users\ASUS\Desktop\ir_project\data_set\Lotte\collection - Copy.jsonl'
        loaded_compressed_index, vectorizer = InitializationQuora()

    results_ID = search_query(query, vectorizer, loaded_compressed_index)
    number_of_results = len(results_ID)
    top_10_results_ID = results_ID[:10]
    results = extract_text_by_ids(Document,top_10_results_ID)

    return results, number_of_results 
# # --------------------------------------------------------------------------------------------------

