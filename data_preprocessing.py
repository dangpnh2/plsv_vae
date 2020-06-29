import sklearn
import pickle
import re
import string
from numpy import random
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


stem = PorterStemmer()
wnl = WordNetLemmatizer()
vectorizer = CountVectorizer(min_df=5)
vectorizer_tfidf = TfidfVectorizer(min_df=5)
stop_words = set(stopwords.words('english')) 

stopwords = ["a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep	keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure	t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero"]
def remove_metadata(lines):
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines

def preprocessing_nonstem(doc):
    doc = doc.lower()
#     doc = remove_metadata(doc)
    
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    
    doc = word_tokenize(doc)
    
    doc = filter(lambda x:x not in string.punctuation, doc)
    
    doc = filter(lambda x:x not in stop_words, doc)
    
    doc = filter(lambda x:not x.isdigit(), doc)
    doc = [wnl.lemmatize(w.lower()) for w in doc]
#     doc = [stem.stem(w) for w in doc]
    doc = ' '.join(e for e in doc)
#     print(doc)
    return doc

def preprocessing(doc):
    doc = doc.lower()
#     doc = remove_metadata(doc)
    
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    
    doc = word_tokenize(doc)
    
    doc = filter(lambda x:x not in string.punctuation, doc)
    
    doc = filter(lambda x:x not in stop_words, doc)
    
    doc = filter(lambda x:not x.isdigit(), doc)
    doc = [wnl.lemmatize(w.lower()) for w in doc]
    doc = [stem.stem(w) for w in doc]
    doc = ' '.join(e for e in doc)
#     print(doc)
    return doc
    
def load_data_20news(subset='all', categories=['soc.religion.christian', 'comp.graphics', 'sci.med'], doc_per_topic=None):
    
    data_train = {}
    
    for cat in categories:
        print('cat: {}'.format(cat))
        fetch_data_train = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42).data
        if doc_per_topic:
            fetch_data_train = fetch_data_train[:doc_per_topic]
        data_train[cat] = [preprocessing(data) for data in fetch_data_train]
    
    
    return data_train

def vectorize_data(data, min_df):
    vectorizer = CountVectorizer(min_df=min_df)
    
    data_flatten = []
    train_label_c = []
    
    
    for cat in data:
        data_flatten+= data[cat]
        train_label_c+= [cat]*len(data[cat])
    
    data_vectorized = vectorizer.fit_transform(data_flatten).toarray()

    return data_vectorized, train_label_c, vectorizer.vocabulary_

def vectorize_data_tfidf(data, min_df):
    vectorizer = TfidfVectorizer(min_df=min_df)
    
    data_flatten = []
    train_label_c = []
    
    
    for cat in data:
        data_flatten+= data[cat]
        train_label_c+= [cat]*len(data[cat])
    
    data_vectorized = vectorizer.fit_transform(data_flatten).toarray()

    return data_vectorized, train_label_c, vectorizer.vocabulary_

def vectorize_data_gensim(preprocessed_data, labels, vocab_keep_n=5000):
#     with open('data/arxiv/preprocessed_text.pkl', 'rb') as f:
#         preprossed_data = pickle.load(f)

#     with open('data/arxiv/preprocessed_labels.pkl', 'rb') as f: 
#             labels = pickle.load(f)        
    labels = [l.split()[1] for l in labels]
    count_lalbes = collections.Counter(labels)
    new_labels = ['astro', 'cs', 'math', 'nucl', 'physics', 'quant', 'stat']
    keep_preprossed_data = []
    keep_labels = []
    for index, label in enumerate(labels):
        if label in new_labels:
            keep_preprossed_data.append(preprossed_data[index])
            keep_labels.append(labels[index])
    train_label = np.array(keep_labels)
    doc_list = keep_preprossed_data; doc_list
    doc_list_split = [doc.split() for doc in doc_list]
    dictionary = corpora.Dictionary(doc_list_split)
    dictionary.filter_extremes(no_below=0, no_above=1,keep_n=vocab_keep_n)
    corpus = np.array([dictionary.doc2bow(line) for line in doc_list_split])
    return corpus, train_label, dictionary, new_labels
