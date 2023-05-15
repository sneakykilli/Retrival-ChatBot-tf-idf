import pandas as pd
from collections import Counter
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# <<<<<<<<<<<<<<<<<<<<<<<< START BLOCK 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Process the Dataframe for use in BOW and tf-idf models
df = pd.read_csv('faq.csv')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
#Tokenize
df['tokenize'] = df["response"].apply(lambda text: nltk.word_tokenize(text))
#Lemmantize
lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text])

df['lemmatized'] = df['tokenize'].apply(lemmatize_words)

#stem
stemmer = PorterStemmer()


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text])

df['text_stemmed'] = df["tokenize"].apply(lambda text: stem_words(text))
df['text_stemmed'] = df["text_stemmed"].apply(lambda text: nltk.word_tokenize(text))
#Remove Stop words
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


df["text_wo_stop"] = df["response"].apply(lambda text: remove_stopwords(text))

#Text to bow_counter
df['text_bow_counter'] = df["text_stemmed"].apply(lambda text: Counter(text))

# <<<<<<<<<<<<<<<<<<<<<<<< END BLOCK 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<< START BLOCK 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Bag of Words Calculations
bow_list = df['text_bow_counter'].tolist()


def compare_overlap(user_message, faq_bow):
    similar_words = 0
    for token in user_message:
        if token in faq_bow:
            similar_words += 1
    return similar_words


def convert_user_message_wordcount(user_message):
    user_message_tok = nltk.word_tokenize(user_message)
    user_message_stem = [stemmer.stem(word) for word in user_message_tok]
    user_bow_count = Counter(user_message_stem)
    return user_bow_count


user_bow_count = convert_user_message_wordcount('Exercise')
df['bow_word_similar'] = df['text_bow_counter'].apply(lambda text: compare_overlap(user_bow_count, text))
max_bow = df['bow_word_similar'].idxmax()
possible_bow_response = df.loc[max_bow, 'response']

# <<<<<<<<<<<<<<<<<<<<<<<< END BLOCK 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<< START BLOCK 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# tf-idf
# Build Corpus of all Documents + user message at [-1]
user_msg = 'Exercise'


class TfChat():

    def tf_idf_response(self, text):
        text = remove_stopwords(text).lower()
        sample_list = [item.lower() for item in df['text_wo_stop'].to_list()]
        preprocessed_docs = [*sample_list, text]
        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform(preprocessed_docs)
        cosine_similarities = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)
        similar_response_index = cosine_similarities.argsort()[0][-2]
        best_response = df['response'].iloc[similar_response_index]
        return best_response

# <<<<<<<<<<<<<<<<<<<<<<<< END BLOCK 3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df.to_csv('doggo_df.csv')


