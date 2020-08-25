import nltk

# NLTK's default stopword list was too extensive and eliminated a lot of useful words
# Hence a self made stopword list.
basic_stopwords_list = {
    'a', 'an', 'all', 'and', 'are', 'as', 'at',
    'be', 'but', 'can', 'do', 'did', 'for',
    'get', 'give', 'has', 'had', 'have', 'how',
    'i', 'if', 'in', 'is', 'it',
    'me', 'my', 'no',
    'of', 'on', 'or',
    'that', 'the', 'there', 'this', 'to', 'up',
    'was', 'we', 'what', 'when', 'why', 'where', 'would', 'with', 'will',
    'you'
}


# This function removes the the stopwords from a given text/string.
# It also strips symbols and converts the text to lower case before verifying it with the basic_stopword_list

def remove_stopwords(status_text):
    symbols = '!"#$%&\'()*+,-./"?:;<=>[\\]^_`{|}~'
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
