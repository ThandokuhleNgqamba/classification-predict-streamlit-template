import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import emoji 


def data_preprocessing(tweet):
    lemm = WordNetLemmatizer()
    #reg = RegexpTokenizer()
    Tokenized_Doc=[]
    stopWords = set(stopwords.words('english'))
    
    tweet = tweet.lower()   
    onlyWords = re.sub('[^a-zA-Z]', ' ', tweet)     #keeping only words
    url = re.compile(r'https?://\S+|www\.\S+')      #removing URLs
    review_ = url.sub(r'',onlyWords)
    html=re.compile(r'<.*?>')
    review_ = html.sub(r'',review_)
    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"     # emoticons
                           u"\U0001F300-\U0001F5FF"     # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"     # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"     # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        
    review_ = emojis.sub(r'',review_)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(review_)
    gen_tweets = [lemm.lemmatize(token) for token in tokens if not token in stopWords]
    Tokenized_Doc.append(gen_tweets)
        #df['tweet tokens'] = pd.Series(Tokenized_Doc)
        
    ans = [' '.join(Tokenized_Doc[0])]
    return ans