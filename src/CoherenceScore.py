import pandas as pd
import re
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
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

def compute_coherence_score_npmi(topic_word_matrix, term_doc_matrix, n_top_words=10, eps=1e-12):
    # Binary doc-term presence
    X = (term_doc_matrix > 0).astype(int)
    N = X.shape[0]  # number of documents

    # Document probabilities
    df = np.array(X.sum(axis=0)).flatten()              # doc freq of each term
    P = df / (N + eps)

    # Precompute co-doc frequencies via sparse ops
    # X.T @ X gives co-occurrence counts (including diagonal)
    C = (X.T @ X).toarray()
    P12 = C / (N + eps)

    scores = []
    for topic in topic_word_matrix:
        top_idx = topic.argsort()[:-n_top_words-1:-1]
        pair_vals = []
        for i in range(len(top_idx)):
            for j in range(i+1, len(top_idx)):
                a, b = top_idx[i], top_idx[j]
                p12 = P12[a, b] + eps
                p1, p2 = P[a] + eps, P[b] + eps
                pmi = np.log(p12 / (p1 * p2))
                npmi = pmi / (-np.log(p12))
                pair_vals.append(npmi)
        scores.append(np.mean(pair_vals) if pair_vals else 0.0)

    return float(np.mean(scores))

def compute_coherence_score(topic_word_matrix, term_doc_matrix):
    """
    Compute the topic coherence score based on word co-occurrence in documents.
    Higher scores indicate more coherent topics.
    """
    n_top_words = 10 
    coherence_scores = []

    # Get document frequencies for terms
    doc_freq = np.array((term_doc_matrix > 0).sum(axis=0)).flatten()

    for topic in topic_word_matrix:
        top_term_indices = topic.argsort()[:-n_top_words-1:-1]
        topic_score = 0
        pairs_count = 0

        # Calculate pairwise scores for top terms
        for i in range(len(top_term_indices)):
            for j in range(i + 1, len(top_term_indices)):
                term1_idx = top_term_indices[i]
                term2_idx = top_term_indices[j]

                # Get co-document frequency
                co_doc_freq = np.sum(np.logical_and(term_doc_matrix[:, term1_idx].toarray(), 
                                                  term_doc_matrix[:, term2_idx].toarray()))

                # Calculate PMI score
                if co_doc_freq > 0:
                    score = np.log((co_doc_freq * term_doc_matrix.shape[0]) / 
                                 (doc_freq[term1_idx] * doc_freq[term2_idx]))
                    topic_score += score
                    pairs_count += 1

        if pairs_count > 0:
            coherence_scores.append(topic_score / pairs_count)
        else:
            coherence_scores.append(0)

    return np.mean(coherence_scores)


def evaluate_topic_models(df, text_column, min_topics=50, max_topics=100, step=10):
    """
    Evaluate LDA models with different numbers of topics and compute their coherence scores.
    """
    # Vectorize text data
    extended_stopwords = stopwords.words('english') + SW_list
    count_vectorizer = CountVectorizer(
        max_df=0.7, 
        min_df=0.0005, 
        stop_words=extended_stopwords
    )
    dtm = count_vectorizer.fit_transform(df[text_column])
    
    coherence_scores = []
    n_topics_range = range(min_topics, max_topics + step, step)
    
    for n_topics in n_topics_range:
        print(f"Evaluating model with {n_topics} topics...")
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            learning_offset=50.,
            random_state=42,
            verbose=1,
            n_jobs=-1,
            max_iter=10,
            doc_topic_prior=0.1,
            topic_word_prior=0.005
        )
        
        lda.fit(dtm)
        coherence = compute_coherence_score_npmi(lda.components_, dtm)
        coherence_scores.append(coherence)
        
        print(f"Coherence score: {coherence}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(n_topics_range), coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Topic Coherence Score vs Number of Topics')
    plt.grid(True)
    plt.savefig('topic_coherence_plot_60T80.png')
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'n_topics': list(n_topics_range),
        'coherence_score': coherence_scores
    })
    results_df.to_csv('topic_coherence_scores_75T90.csv', index=False)
    
    return results_df

df = pd.read_csv('ATCM_IPWP_WithText_PlusScientific.csv', low_memory = False)
df['text'] = df['text'].apply(lambda x: preprocess_text(x))

results = evaluate_topic_models(df, 'text', min_topics=75, max_topics=90, step=5)
