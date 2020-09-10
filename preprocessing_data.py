import nltk
import re

# NLTK's default stopword list was too extensive and eliminated a lot of useful words
# Hence a self made stopword list.
basic_stopwords_list = {
    'a', 'am', 'an', 'all', 'and', 'are', 'as', 'at',
    'be', 'but', 'can', 'do', 'did', 'for',
    'get', 'give', 'has', 'had', 'have', 'how', 'he',
    'i', 'if', 'in', 'is', 'it',
    'me', 'my', 'no',
    'of', 'on', 'or', 'our', 'she',
    'that', 'the', 'there', 'this', 'to', 'up',
    'was', 'we', 'what', 'when', 'why', 'where', 'would', 'with', 'will',
    'you'
}


# This function removes the the stopwords from a given text/string.
# It also strips symbols and converts the text to lower case before verifying it with the basic_stopword_list

def remove_stopwords(status_text):
    symbols = '!"#$%&\'()*+,-./"?:;<=>[\\]^_`{|}~'
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    status_text = re.sub(text_cleaning_re, ' ', str(status_text).lower()).strip()
    stopword_removed_status_text = []
    for word in status_text.lower().split():
        if word.strip(symbols) not in basic_stopwords_list:
            if word.strip(symbols) != '' or ' ':
                stopword_removed_status_text.append(word.strip(symbols))
    full_string = ' '.join(stopword_removed_status_text)
    return full_string


# The function lemmatizes the texts, that is breaks words into its simplest form.
# This gives better results over stemming in my experience.
def lemmatize_status(status):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_status = []
    for word in status.split():
        lemmatized_status.append(lemmatizer.lemmatize(word))
    full_string = ' '.join(lemmatized_status)
    return full_string


# The get_clean_text method calls the above two functions and processes the text received.
def get_clean_text(status_text):
    join_text = []
    clean_text = lemmatize_status(remove_stopwords(status_text))
    for word in clean_text.split():
        if len(word) > 2:
            join_text.append(word)
    full_string = ' '.join(join_text)
    return full_string
