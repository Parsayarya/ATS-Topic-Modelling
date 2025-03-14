import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pandas as pd
import re
from scipy.sparse import csr_matrix  # Import sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

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



df = pd.read_csv('Data/Input/merged_Corpus_1961-2023.csv')
df['text'] = df['text'].apply(lambda x : preprocess_text(x))
print('here')

def monitor_convergence(df, text_column, n_topics=75, learning_offset=50.0, max_iter=3):
    """
    Monitor LDA model convergence with different iterations and learning offset
    """
    extended_stopwords = stopwords.words('english') + SW_list
    count_vectorizer = CountVectorizer(
        max_df=0.7,
        min_df=0.005,
        stop_words=extended_stopwords
    )
    dtm = count_vectorizer.fit_transform(df[text_column])
    
    # Store perplexity scores for each iteration
    perplexity_scores = []
    log_likelihood_scores = []
    
    # Fit LDA and monitor convergence
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        learning_offset=learning_offset,
        random_state=42,
        verbose=1,
        n_jobs=-1,
        max_iter=max_iter,
        doc_topic_prior=0.1,
        topic_word_prior=0.005
    )
    
    for i in range(max_iter):
        # Partial fit for each iteration
        print(i)
        lda.partial_fit(dtm)
        perplexity_scores.append(lda.perplexity(dtm))
        log_likelihood_scores.append(lda.score(dtm))
        
    # Plot convergence metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot perplexity
    ax1.plot(range(1, max_iter + 1), perplexity_scores, marker='o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity vs Iteration')
    ax1.grid(True)
    
    # Plot log-likelihood
    ax2.plot(range(1, max_iter + 1), log_likelihood_scores, marker='o', color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood vs Iteration')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('lda_convergence.png')
    plt.close()
    
    # Calculate convergence statistics
    perplexity_changes = np.diff(perplexity_scores)
    likelihood_changes = np.diff(log_likelihood_scores)
    
    convergence_stats = pd.DataFrame({
        'iteration': range(1, max_iter+1),
        'perplexity_scores': perplexity_scores,
        'log_likelihood_scores': log_likelihood_scores,
        'relative_perplexity_change': perplexity_changes / np.abs(perplexity_scores[:-1]),
        'relative_likelihood_change': likelihood_changes / np.abs(log_likelihood_scores[:-1])
    })
    
    convergence_stats.to_csv('convergence_stats.csv', index=False)
    
    return convergence_stats


stats = monitor_convergence(df, 'text', learning_offset=50.0)
# Test different learning offsets
# learning_offsets = [30.0, 50.0, 70.0]
# offset_results = {}

# for offset in learning_offsets:
#     stats = monitor_convergence(df, 'text', learning_offset=offset)
#     offset_results[offset] = stats['relative_likelihood_change'].mean()

# # Plot learning offset comparison
# plt.figure(figsize=(8, 5))
# plt.plot(learning_offsets, [offset_results[o] for o in learning_offsets], marker='o')
# plt.xlabel('Learning Offset')
# plt.ylabel('Average Relative Likelihood Change')
# plt.title('Impact of Learning Offset on Convergence')
# plt.grid(True)
# plt.savefig('learning_offset_comparison.png')
# plt.close()