import pandas as pd
import re
from scipy.sparse import csr_matrix  # Import sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
SW_list = ['al', 'et', 'mt', 'ip', 'ofthe', 'ix', 'xxiii', 'use','used','february', 'january', 'december',
           'march', 'november' , 'october','atcm', 'including','iii','set','view','party','annex','year',
           'government','kingdom','accordance','submitted', 'chile','located','small','contact','available'
           ,'report','document','meeting','secretariat','key','new','zealand','land','united','level','shall',
           'accordance','person','antarctica','purpose','material','description','ensure','issue','based','question',
           'need','initial','effort','understanding','work','using','carried','noted','source', 'uk','year','co','also','de',
           'ii','le','iv', 'would', 'could', 'may', 'wether', 'might', 'however', 'paragraph','reported','within','inthe', 'one', 'first',
           'table', 'figure','due','two', 'per', 'total', 'day', 'whether', 'consider', 'provided', 'case', 'taken', 'well',
           'part','time','made','form', 'from', 'subject', 're', 'edu', 'use', 'el', 'nio', 'three', 'two', 'also', 'since', 'using', 'thu', 'nh','de', 'en', 'la', 'del', 'se', 'sur', 'un', 'sar', 'mi', 'son', 'si', 'related', 'based', 'from', 'tc', 'sc', 'bob', 'io', 'wp', 'mar', 'mr', 'hut',
           'nao', 'eof', 'cp', 'ep', 'te', 'ce', 'ar', 'dic', 'amo', 'sic', 'ie', 'ae', 'th', 'sat', 'atthe', 'atm', 'thatthe', 'onthe', 'ad', 'ea','nd', 'ss', 'clade', 'tp', 'ant', 'tho', 'doc', 'tratado', 'dom', 'cf', 'ohc', 'item', 'sn', 'hf', 'ac', 'le', 'ssa', 'cn',
           'los','sam', 'paragraph', 'several', 'although', 'ofthe', 'inthe', 'tothe', 'forthe', 'et al', 'al', 'et',
           'ch','pg','wt','mp', 'ee', 'sie', 'andor', 'ta', 'om', 'ot', 'na', 'oa', 'mo', 'jan', 'ba', 'ree', 'ia', 'li', 'ne',
           'sa', 'fr', 'aod', 'mids', 'fo', 'ap', 'pom', 'hc', 'il', 'rom', 'fa', 'ind', 'ate', 'asa', 'sla', 'ri', 'msa', 'ofa', 'eel',
           'ec', 'ssh', 'ao', 'pc', 'ra', 'np', 'persian', 'oman', 'da', 'pf', 'que', 'ml', 'sio', 'sm', 'ci', 'nta', 'em', 'meander', 'mcdw', 'ray',
           'may', 'tio', 'mld', 'theantarctic', 'itis', 'ti', 'wil', 'withthe', 'tobe', 'andthe', 'tt', 'wo', 'iv', 'sys', 'pp',
           'st', 'po', 'atsea', 'aden', 'tg', 'tl', 'par', 'gm', 'ww', 'sb', 'ct', 'ion', 'lat', 'ft', 'sag', 'mt', 'sec', 'ho',
           'ai', 'insitu', 'aa', 'pa', 'ic', 'sl', 'ro', 'tn', 'pe', 'lig', 'oe', 'ed', 'pu','tna', 'thet',
           'sp', 'fe', 'nov', 'eocene', 'sw', 'ni', 'mn', 'nw', 'ird', 'por', 'npo', 'maastrichtian', 'para', 'new', 'herein',
            'ag', 'ste', 'toa', 'pan', 'nt', 'ol', 'ke', 'sit', 'spi', 'eg', 'br', 'arca', 'bythe', 'ina', 'ef', 'sol', 'ge', 'fromthe', 'etal',
           'oc', 'cam', 'va', 'upb', 'gom', 'ace', 'eke', 'ond', 'ny', 'lia', 'er', 'ase', 'som', 'sf', 'vanda', 'sah', 'isw', 'bo', 'aegean',
           'tine', 'co', 'fron', 'op', 'vith', 'sone', 'thie', 'vhich', 'thc', 'say','isthe', 'isa', 'othe','shouldbe', 'wa', 'ther', 'page', 'maine'
           'du', 'su', 'ha', 'con', 'mme', 'fin','bight', 'ga', 'nn', 'ab', 'dsw', 'dp', 'lr', 'ln', 'soc',
           'fom', 'pr', 'ing', 'tee', 'cr', 'fst', 'sd', 'tha', 'oi', 'ela','ow', 'wi', 'tat',
           'md', 'ca', 'rev', 'ty', 'engl', 'aw','bi','tm','tb', 'lw', 'cat', 'ew', 'lee', 'ite', 'af', 'seta', 'lf', 'tex', 'ete', 'snp', 'bl', 'asc', 'sep', 'nb', 'ay', 'rh', 'fp',
           'im', 'pi', 'au', 'end', 'mca', 'hr', 'tht', 'ofthese', 'wes', 'wih', 'twc', 'wc', 'lt', 'hat', 'tom', 'world',
           'san', 'mc', 'cc', 'dfe', 'baja', 'costa', 'instituto', 'isla', 'antrtico', 'tail', 'lm', 'ei',
           'din','oo', 'ig', 'song', 'mh','eo', 'oni','toe', 'gi','gam',
           'zealand', 'uk', 'chile', 'kingdom', 'argentina', 'argentine', 'british', 'antarctica', 'france', 'february', 'january', 'netherlands', 'list', 'feb',
           'within', 'antarctic',
           'china', 'indian', 'arabian', 'india', 'asian','european', 'alaska', 'bengal',
           'arctic', 'greenland', 'europe', 'norwegian', 'canada', 'australian',
            'brazil', 'brazilian','peru', 'non',
           'australia', 'russian',
           'japan', 'japanese',
           'june', 'october', 'july', 'september', 'august', 'portugal',
           'africa','african','iran', 'mozambique',
           'berlin', 'basel',  'antarctictreaty', 'antartica',  'antaretic', 'ofthis',  'tibetan', 'inantarctica',
           'year', 'summer', 'season', 'day', 'south', 'winter', 'month','december', 'march', 'austral','america',
           'found', 'vi', 'vii', 'ltd', 'llc', 'inc', 'dr', 'via', 'nm', 'taylor', 'francis', 'forall', 'ob', 'du', 'taato', 'laato', 'ato', 'lato', 'aato', 'ofantarctica',
           'fax', 'mail', 'aire', 'buenos', 'base', 'usa', 'naval', 'opening', 'ecuador', 'punta', 'martin', 'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica', 'argentino', 'jorge', 'tel', 'tierra',
           'buenos', 'base', 'usa', 'naval', 'opening', 'ecuador', 'punta', 'martin', 'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica',
           'argentino', 'jorge', 'tel', 'tierra', 'dio', 'lo'
           'opening', 'ecuador', 'punta', 'martin', 'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica','martin',
            'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica','uruguay','nacional',
           'antrtica', 'argentino', 'jorge', 'tel', 'tierra', 'italy', 'franch', 'ye', 'ou', 'pre','xxxviii','xxxix',  'ana', 'ane', 'ote', 'ue', 'ene', 'wth', 'ry', 'eee', 'ear', 'aay', 'sue', 'wich', 'ara', 'og', 'tae', 'tr', 'ey', 'dat', 'treatyconsultative', 'inte', 'tion', 'ary', 'bt', 'tne', 'ater'
           , 'xx', 'xi', 'viii', 'xliv', 'xlii', 'asthe', 'pat', 'ean', 'jc', 'jp', 'willbe', 'yes', 'yet', 'spain', 'date', 'belgian', 'bulgarian', 'turkey', 'bulgaria', 'belgium',
           'turkish', 'antrtida', 'italian', 'inthis', 'shal', 'staton', 'seo', 'sr', 'korea', 'korean', 'numberof', 'isnot', 'ation', 'andlor', 'km', 'art', 'lo', 'av', 'ser', 'rea', 'sus', 'este',
           'heen', 'von', 'slower','mam','pre', 'mv',"xviii", "cal", "opal", "sill", "reunion", "toc", "cone", "tem", "cutoff", "lp", "subgroup", "nat", "eep", "ast", "bulletin",
           "pro", "pig", "rigidity", "psa", "lst", "oh", "pt", "gl", "predominant",
           "tec", "inhibited", "pm", "arar", "miz", "eps", "lowtemperature", "setup", "gw", "wv", "ssi", "ir", "anita", "monte",
           "sod", "ut", "pwv", "tor",
           "sia", "rsl", "rv", "hg", "dm", "ka", "rise", "model", "basal", "retreat", "bp", "aq", "reader", "ending", "wileyvch",
           "germany", "note", "attached", "norway", "sweden", "poland", "xvi", "transcript", "xiii", "finland", "prof", "belgica",
           "entered", "david", "following", "pei", "romania", "title", "gev", "midnight", "colombia", "britain", "swir",
           "forty", "aaiw", "nadw",
           "qbo", "psu", "switzerland", "psc", "bc", "informa", "log", 
           "rms", "cv", "lay", "swedish", "rd", "bar", "anion", "accord", "theprotocol",
           "iso", "bb", "zr", "id", "sao", 
           "many", "possible", "problem", "even", "present", "therefore", "process", "thus", "make", "example", "condition",
           "must", "still", "often", "given", "considered", "way", "particular", "order", "place", "number", "large", "point", "much"

 ]

def preprocess_text(text: str) -> str:
    """
    Preprocess a given text by removing punctuation, special characters, digits,
    and then lemmatizing all the words.

    Parameters:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.lower()
    tokens = nltk.word_tokenize(cleaned_text)
    extended_stopwords = stopwords.words('english') + SW_list
    tokens = [word for word in tokens if word not in extended_stopwords and len(word) > 1]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text
def preprocess_text2(text: str, use_stemming: bool = False) -> str:
    """
    Preprocess a given text by removing punctuation, special characters, digits,
    converting to lowercase, removing stopwords, and then lemmatizing (and optionally stemming) all the words.

    Parameters:
    text (str): The input text to be preprocessed.
    use_stemming (bool): If true, apply stemming after lemmatization.

    Returns:
    str: The preprocessed text.
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = nltk.word_tokenize(cleaned_text)
    extended_stopwords = stopwords.words('english') + SW_list
    tokens = [word for word in tokens if word not in extended_stopwords and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer() if use_stemming else None
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    if use_stemming:
        lemmatized_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

    # Join tokens back into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text



def LDAV1(df, text_column, n_topics=30, n_top_words=25):
    """
    Perform LDA topic modeling on a given DataFrame and extract topic scores and top words for each topic.

    :param df: Input pandas DataFrame containing text data.
    :param text_column: The name of the column in the DataFrame containing text data.
    :param n_topics: Number of topics to model (default 30).
    :param n_top_words: Number of top words to extract for each topic (default 25).
    :return: Tuple of DataFrames (topic_scores_df, topics_df) where:
             - topic_scores_df contains original data with added topic score columns,
             - topics_df lists top words for each topic.
    """
    # Vectorize text data using sparse matrix
    extended_stopwords = stopwords.words('english') + SW_list
    count_vectorizer = CountVectorizer(max_df=0.7, min_df=0.0005, stop_words=extended_stopwords) #TODO: max_df and min_df(min_df is too low(go for 0.005, it was 0.002 for the 90 version))
    dtm: csr_matrix = count_vectorizer.fit_transform(df[text_column])  # Ensure DTM is sparse
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', learning_offset=50., random_state=42,
                                    verbose=1, n_jobs=14, max_iter=10, doc_topic_prior=0.1, topic_word_prior=0.005).fit(dtm) #TODO: Change Max_iter to be lower thaan 10 this time

    tf_feature_names = count_vectorizer.get_feature_names_out()
    topics_df = pd.DataFrame(index=range(n_top_words))
    for topic_idx, topic in enumerate(lda.components_):
        topics_df[f"Topic {topic_idx}"] = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    topic_scores = lda.transform(dtm)
    for i in range(n_topics):
        df[f"Topic {i} Score"] = topic_scores[:, i]

    df.drop(columns=[text_column], inplace=True)

    return df, topics_df

df = pd.read_csv('Data/Input/merged_Corpus_1961-2023.csv')
# print(df.Type.value_counts())
# df = df[(df['Type'] == 'ip') | (df['Type'] == 'wp')]
df['text'] = df['text'].apply(lambda x : preprocess_text(x))
df2, topics_df = LDAV1(df, 'text', n_topics=100, n_top_words=25)
df2.to_csv('Data/Output2/100_LDA_Topic_Loading_ScoreV2.csv',index=False)
topics_df.to_csv('Data/Output2/100_LDA_Topic_WordsV2.csv', index=False)

# ----------------------------------------------------------------------------------- The one to sendout
import pandas as pd
import re
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
import numpy as np

SW_list = ['al', 'et', 'mt', 'ip', 'ofthe', 'ix', 'xxiii', 'use','used','february', 'january', 'december',
           'march', 'november' , 'october','atcm', 'including','iii','set','view','party','annex','year',
           'government','kingdom','accordance','submitted', 'chile','located','small','contact','available'
           ,'report','document','meeting','secretariat','key','new','zealand','land','united','level','shall',
           'accordance','person','antarctica','purpose','material','description','ensure','issue','based','question',
           'need','initial','effort','understanding','work','using','carried','noted','source', 'uk','year','co','also','de',
           'ii','le','iv', 'would', 'could', 'may', 'wether', 'might', 'however', 'paragraph','reported','within','inthe', 'one', 'first',
           'table', 'figure','due','two', 'per', 'total', 'day', 'whether', 'consider', 'provided', 'case', 'taken', 'well',
           'part','time','made','form', 'from', 'subject', 're', 'edu', 'use', 'el', 'nio', 'three', 'two', 'also', 'since', 'using', 'thu', 'nh','de', 'en', 'la', 'del', 'se', 'sur', 'un', 'sar', 'mi', 'son', 'si', 'related', 'based', 'from', 'tc', 'sc', 'bob', 'io', 'wp', 'mar', 'mr', 'hut',
           'nao', 'eof', 'cp', 'ep', 'te', 'ce', 'ar', 'dic', 'amo', 'sic', 'ie', 'ae', 'th', 'sat', 'atthe', 'atm', 'thatthe', 'onthe', 'ad', 'ea','nd', 'ss', 'clade', 'tp', 'ant', 'tho', 'doc', 'tratado', 'dom', 'cf', 'ohc', 'item', 'sn', 'hf', 'ac', 'le', 'ssa', 'cn',
           'los','sam', 'paragraph', 'several', 'although', 'ofthe', 'inthe', 'tothe', 'forthe', 'et al', 'al', 'et',
           'ch','pg','wt','mp', 'ee', 'sie', 'andor', 'ta', 'om', 'ot', 'na', 'oa', 'mo', 'jan', 'ba', 'ree', 'ia', 'li', 'ne',
           'sa', 'fr', 'aod', 'mids', 'fo', 'ap', 'pom', 'hc', 'il', 'rom', 'fa', 'ind', 'ate', 'asa', 'sla', 'ri', 'msa', 'ofa', 'eel',
           'ec', 'ssh', 'ao', 'pc', 'ra', 'np', 'persian', 'oman', 'da', 'pf', 'que', 'ml', 'sio', 'sm', 'ci', 'nta', 'em', 'meander', 'mcdw', 'ray',
           'may', 'tio', 'mld', 'theantarctic', 'itis', 'ti', 'wil', 'withthe', 'tobe', 'andthe', 'tt', 'wo', 'iv', 'sys', 'pp',
           'st', 'po', 'atsea', 'aden', 'tg', 'tl', 'par', 'gm', 'ww', 'sb', 'ct', 'ion', 'lat', 'ft', 'sag', 'mt', 'sec', 'ho',
           'ai', 'insitu', 'aa', 'pa', 'ic', 'sl', 'ro', 'tn', 'pe', 'lig', 'oe', 'ed', 'pu','tna', 'thet',
           'sp', 'fe', 'nov', 'eocene', 'sw', 'ni', 'mn', 'nw', 'ird', 'por', 'npo', 'maastrichtian', 'para', 'new', 'herein',
            'ag', 'ste', 'toa', 'pan', 'nt', 'ol', 'ke', 'sit', 'spi', 'eg', 'br', 'arca', 'bythe', 'ina', 'ef', 'sol', 'ge', 'fromthe', 'etal',
           'oc', 'cam', 'va', 'upb', 'gom', 'ace', 'eke', 'ond', 'ny', 'lia', 'er', 'ase', 'som', 'sf', 'vanda', 'sah', 'isw', 'bo', 'aegean',
           'tine', 'co', 'fron', 'op', 'vith', 'sone', 'thie', 'vhich', 'thc', 'say','isthe', 'isa', 'othe','shouldbe', 'wa', 'ther', 'page', 'maine'
           'du', 'su', 'ha', 'con', 'mme', 'fin','bight', 'ga', 'nn', 'ab', 'dsw', 'dp', 'lr', 'ln', 'soc',
           'fom', 'pr', 'ing', 'tee', 'cr', 'fst', 'sd', 'tha', 'oi', 'ela','ow', 'wi', 'tat',
           'md', 'ca', 'rev', 'ty', 'engl', 'aw','bi','tm','tb', 'lw', 'cat', 'ew', 'lee', 'ite', 'af', 'seta', 'lf', 'tex', 'ete', 'snp', 'bl', 'asc', 'sep', 'nb', 'ay', 'rh', 'fp',
           'im', 'pi', 'au', 'end', 'mca', 'hr', 'tht', 'ofthese', 'wes', 'wih', 'twc', 'wc', 'lt', 'hat', 'tom', 'world',
           'san', 'mc', 'cc', 'dfe', 'baja', 'costa', 'instituto', 'isla', 'antrtico', 'tail', 'lm', 'ei',
           'din','oo', 'ig', 'song', 'mh','eo', 'oni','toe', 'gi','gam',
           'zealand', 'uk', 'chile', 'kingdom', 'argentina', 'argentine', 'british', 'antarctica', 'france', 'february', 'january', 'netherlands', 'list', 'feb',
           'within', 'antarctic',
           'china', 'indian', 'arabian', 'india', 'asian','european', 'alaska', 'bengal',
           'arctic', 'greenland', 'europe', 'norwegian', 'canada', 'australian',
            'brazil', 'brazilian','peru', 'non',
           'australia', 'russian',
           'japan', 'japanese',
           'june', 'october', 'july', 'september', 'august', 'portugal',
           'africa','african','iran', 'mozambique',
           'berlin', 'basel',  'antarctictreaty', 'antartica',  'antaretic', 'ofthis',  'tibetan', 'inantarctica',
           'year', 'summer', 'season', 'day', 'south', 'winter', 'month','december', 'march', 'austral','america',
           'found', 'vi', 'vii', 'ltd', 'llc', 'inc', 'dr', 'via', 'nm', 'taylor', 'francis', 'forall', 'ob', 'du', 'taato', 'laato', 'ato', 'lato', 'aato', 'ofantarctica',
           'fax', 'mail', 'aire', 'buenos', 'base', 'usa', 'naval', 'opening', 'ecuador', 'punta', 'martin', 'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica', 'argentino', 'jorge', 'tel', 'tierra',
           'buenos', 'base', 'usa', 'naval', 'opening', 'ecuador', 'punta', 'martin', 'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica',
           'argentino', 'jorge', 'tel', 'tierra', 'dio', 'lo'
           'opening', 'ecuador', 'punta', 'martin', 'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica','martin',
            'address', 'esperanza', 'sobre', 'carlos', 'uruguay', 'nacional', 'antrtica','uruguay','nacional',
           'antrtica', 'argentino', 'jorge', 'tel', 'tierra', 'italy', 'franch', 'ye', 'ou', 'pre','xxxviii','xxxix',  'ana', 'ane', 'ote', 'ue', 'ene', 'wth', 'ry', 'eee', 'ear', 'aay', 'sue', 'wich', 'ara', 'og', 'tae', 'tr', 'ey', 'dat', 'treatyconsultative', 'inte', 'tion', 'ary', 'bt', 'tne', 'ater'
           , 'xx', 'xi', 'viii', 'xliv', 'xlii', 'asthe', 'pat', 'ean', 'jc', 'jp', 'willbe', 'yes', 'yet', 'spain', 'date', 'belgian', 'bulgarian', 'turkey', 'bulgaria', 'belgium',
           'turkish', 'antrtida', 'italian', 'inthis', 'shal', 'staton', 'seo', 'sr', 'korea', 'korean', 'numberof', 'isnot', 'ation', 'andlor', 'km', 'art', 'lo', 'av', 'ser', 'rea', 'sus', 'este',
           'heen', 'von', 'slower','mam','pre', 'mv'
 ]

def preprocess_text(text: str) -> str:

    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespace
    cleaned_text = cleaned_text.lower().strip()
    
    tokens = nltk.word_tokenize(cleaned_text)
    extended_stopwords = stopwords.words('english') + SW_list
    
    tokens = [word for word in tokens if word not in extended_stopwords and len(word) > 2]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(lemmatized_tokens)

def LDAV2(df, text_column, n_topics=75, n_top_words=20):
    extended_stopwords = stopwords.words('english') + SW_list
    count_vectorizer = CountVectorizer(
        max_df=0.7,  # Lowered from 0.9 to remove more common terms
        min_df=0.001,  # Increased from 0 to remove rare terms
        stop_words=extended_stopwords,
    )
    
    dtm: csr_matrix = count_vectorizer.fit_transform(df[text_column])
    
    # Enhanced LDA parameters
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        learning_offset=50.,
        random_state=42,
        verbose=1,
        n_jobs=-1,
        max_iter=15,  # Slightly increased for better convergence
        doc_topic_prior=0.1,  # Lower alpha for more specific topic assignments
        topic_word_prior=0.01,  # Lower beta for more specific word-topic assignments
        batch_size=128  # Smaller batch size for better learning
    )
    
    lda.fit(dtm)
    
    tf_feature_names = count_vectorizer.get_feature_names_out()
    topics_df = pd.DataFrame(index=range(n_top_words))
    topic_word_probs = pd.DataFrame(index=range(n_top_words))
    
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-n_top_words-1:-1]
        topics_df[f"Topic {topic_idx}"] = [tf_feature_names[i] for i in top_word_indices]
        topic_word_probs[f"Topic {topic_idx} Prob"] = [topic[i] for i in top_word_indices]
    
    # Calculate topic scores
    topic_scores = lda.transform(dtm)
    
    for i in range(n_topics):
        df[f"Topic {i} Score"] = topic_scores[:, i]
    
    
    df.drop(columns=[text_column], inplace=True)
    
    return df, topics_df, topic_word_probs


df = pd.read_csv('Data/Input/merged_Corpus_1961-2023.csv')
df['text'] = df['text'].apply(lambda x: preprocess_text(x))
df2, topics_df, topic_probs = LDAV2(df, 'text', n_topics=75, n_top_words=25)

df2.to_csv('Data/Output/refined/75_LDA_Topic_Loading_Score_V2.csv', index=False)
topics_df.to_csv('Data/Output/refined/75_LDA_Topic_Words_V2.csv', index=False)
topic_probs.to_csv('Data/Output/refined/75_LDA_Topic_Word_Probabilities_V2.csv', index=False)
#c----------------------------------------------------------------------------------------------------- TFIDF
