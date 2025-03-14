import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from typing import Dict, List, Tuple
import re

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
# Threat-specific seed words
THREAT_SEEDS = {
    "climate_terrestrial": [
        "temperature", "warming", "precipitation", "permafrost", "glacier", 
        "terrestrial", "ecosystem", "vegetation", "soil", "ice"
    ],
    "climate_marine": [
        "sea", "ice", "marine", "ocean", "current", "circulation", "krill",
        "fishery", "penguin", "seal"
    ],
    "ocean_acidification": [
        "acidification", "ph", "carbonate", "calcium", "shell", "coral",
        "dissolution", "aragonite", "calcification", "plankton"
    ],
    "invasive_species": [
        "invasive", "alien", "species", "pathogen", "disease", "introduced",
        "non","native", "biosecurity", "pest", "invasion"
    ],
    "mineral_exploitation": [
        "mining", "mineral", "hydrocarbon", "oil", "gas", "exploration",
        "drilling", "resource", "extraction", "deposit"
    ],
    "bioprospecting": [
        "bioprospecting", "genetic", "resource", "pharmaceutical", "enzyme",
        "organism", "biodiversity", "commercial", "patent", "biological"
    ],
    "pollution": [
        "pollution", "waste", "contamination", "plastic", "chemical",
        "microplastic", "sewage", "oil", "fuel", "debris"
    ],
    "tourism_impacts": [
        "tourism", "visitor", "impact", "station", "facility", "footprint",
        "scientific", "vessel", "aircraft", "infrastructure"
    ],
    "geopolitical": [
        "political", "territorial", "claim", "sovereignty", "dispute",
        "international", "cooperation", "conflict", "nation", "interest"
    ],
    "governance": [
        "treaty", "protocol", "governance", "regulation", "management",
        "policy", "compliance", "measure", "implementation", "enforcement"
    ]
}

class DocumentPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.extended_stopwords = stopwords.words('english') + SW_list
        
    def preprocess(self, text: str) -> str:
        """Basic preprocessing with lemmatization"""
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.lower().strip()
        tokens = word_tokenize(cleaned_text)
        tokens = [word for word in tokens if word not in self.extended_stopwords and len(word) > 2]
        return ' '.join([self.lemmatizer.lemmatize(word) for word in tokens])

class TopicModelBase:
    """Base class for topic modeling approaches"""
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.preprocessor = DocumentPreprocessor()
        self.vectorizer = None
        self.lda_model = None
        
    def _create_vectorizer(self, use_tfidf: bool = False):
        """Create appropriate vectorizer"""
        params = {
            'max_df': 0.7,
            'min_df': 0.01,
            'max_features': 10000,
            'stop_words': self.preprocessor.extended_stopwords
        }
        return TfidfVectorizer(**params) if use_tfidf else CountVectorizer(**params)
    
    def _create_lda_model(self):
        """Create LDA model with standard parameters"""
        return LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_offset=50.,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
            max_iter=10,
            verbose=1,
            learning_method='online',
            # batch_size=128,
            random_state=42,
            n_jobs=-1
        )
    
    def calculate_coherence(self, topic_words: List[str], dtm: csr_matrix) -> float:
        """Calculate topic coherence using NPMI"""
        word_indices = [self.vectorizer.vocabulary_[word] 
                       for word in topic_words 
                       if word in self.vectorizer.vocabulary_]
        
        if len(word_indices) < 2:
            return 0.0
        
        doc_freq = np.array(dtm.sum(axis=0))[0]
        n_docs = dtm.shape[0]
        
        npmi_scores = []
        for i, w1_idx in enumerate(word_indices[:-1]):
            for w2_idx in word_indices[i+1:]:
                co_freq = dtm[:, w1_idx].multiply(dtm[:, w2_idx]).sum()
                
                if co_freq == 0:
                    continue
                
                p_w1 = doc_freq[w1_idx] / n_docs
                p_w2 = doc_freq[w2_idx] / n_docs
                p_both = co_freq / n_docs
                
                pmi = np.log(p_both / (p_w1 * p_w2))
                npmi = pmi / (-np.log(p_both))
                npmi_scores.append(npmi)
        
        return np.mean(npmi_scores) if npmi_scores else 0.0
    
    def _create_results(self, df: pd.DataFrame ,topic_scores: np.ndarray, dtm: csr_matrix) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create standardized results dataframes"""
        topic_words_df = pd.DataFrame(
            {f"Topic {i}": [self.vectorizer.get_feature_names_out()[idx] 
                           for idx in topic.argsort()[:-20-1:-1]]
             for i, topic in enumerate(self.lda_model.components_)}
        )
        
        # Calculate coherence scores
        coherence_scores = []
        for topic_idx in range(self.n_topics):
            topic_terms = topic_words_df[f"Topic {topic_idx}"].tolist()
            coherence = self.calculate_coherence(topic_terms, dtm)
            coherence_scores.append(coherence)
        
        topic_words_df.loc["coherence"] = coherence_scores
        
        # Create topic scores DataFrame
        scores_df = pd.DataFrame(
            topic_scores,
            columns=[f"Topic {i} Score" for i in range(self.n_topics)]
        )
        
        # Include original columns from df
        scores_df = pd.concat([df[['Type', 'Year', 'Title']].reset_index(drop=True), scores_df], axis=1)
        
        return scores_df, topic_words_df
    
class GuidedLDAApproach(TopicModelBase):
    """Guided LDA with threat-specific seed words"""
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Prepare training sample
        policy_docs = df[df['Type'].isin(['ip', 'wp'])]
        scientific_docs = df[df['Type'] == 'Scientific'].sample(
            n=len(policy_docs)*2,
            random_state=42
        )
        training_df = pd.concat([policy_docs, scientific_docs])
        
        # Preprocess all documents
        all_processed = df['text'].apply(self.preprocessor.preprocess)
        training_processed = training_df['text'].apply(self.preprocessor.preprocess)
        
        # Create and fit vectorizer on training data
        self.vectorizer = self._create_vectorizer()
        training_dtm = self.vectorizer.fit_transform(training_processed)
        
        # Initialize and train LDA model
        self.lda_model = self._create_lda_model()
        self.lda_model.components_ = self._initialize_topic_word_matrix(
            self.vectorizer.vocabulary_
        )
        self.lda_model.fit(training_dtm)
        
        # Transform full corpus
        full_dtm = self.vectorizer.transform(all_processed)
        topic_scores = self.lda_model.transform(full_dtm)
        
        return self._create_results(df ,topic_scores, full_dtm)
    
    def _initialize_topic_word_matrix(self, vocabulary: Dict[str, int]) -> np.ndarray:
        """Initialize topic-word matrix with seed words"""
        n_words = len(vocabulary)
        beta = np.ones((self.n_topics, n_words)) * 0.01
        
        for topic_idx, (threat, seeds) in enumerate(THREAT_SEEDS.items()):
            for seed in seeds:
                if seed in vocabulary:
                    beta[topic_idx, vocabulary[seed]] = 1.0
        
        return beta / beta.sum(axis=1)[:, np.newaxis]

class DomainEnhancedApproach(TopicModelBase):
    """Enhanced preprocessing with domain-specific terms"""
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Prepare training sample
        policy_docs = df[df['Type'].isin(['ip', 'wp'])]
        scientific_docs = df[df['Type'] == 'Scientific'].sample(
            n=len(policy_docs)*2,
            random_state=42
        )
        training_df = pd.concat([policy_docs, scientific_docs])
        
        # Enhanced preprocessing
        all_processed = df['text'].apply(self._enhance_text)
        training_processed = training_df['text'].apply(self._enhance_text)
        
        # Create and fit vectorizer
        self.vectorizer = self._create_vectorizer(use_tfidf=True)
        training_dtm = self.vectorizer.fit_transform(training_processed)
        
        # Train LDA model
        self.lda_model = self._create_lda_model()
        self.lda_model.fit(training_dtm)
        
        # Transform full corpus
        full_dtm = self.vectorizer.transform(all_processed)
        topic_scores = self.lda_model.transform(full_dtm)
        
        return self._create_results(df, topic_scores, full_dtm)
    
    def _enhance_text(self, text: str) -> str:
        """Enhance text by repeating threat-related terms"""
        processed = self.preprocessor.preprocess(text)
        tokens = processed.split()
        enhanced_tokens = []
        
        for token in tokens:
            weight = 1
            for threat_seeds in THREAT_SEEDS.values():
                if token in threat_seeds:
                    weight = 3
                    break
            enhanced_tokens.extend([token] * weight)
        
        return ' '.join(enhanced_tokens)

class DocumentFilteringApproach(TopicModelBase):
    """Filter documents based on threat relevance"""
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Calculate threat relevance for all documents
        df['threat_relevance'] = df['text'].apply(self._calculate_threat_relevance)
        
        # Select documents for training
        threshold = df['threat_relevance'].quantile(0.7)
        relevant_docs = df[df['threat_relevance'] >= threshold]
        
        policy_docs = relevant_docs[relevant_docs['Type'].isin(['ip', 'wp'])]
        scientific_docs = relevant_docs[relevant_docs['Type'] == 'Scientific'].sample(
            n=min(len(policy_docs)*2, len(relevant_docs[relevant_docs['Type'] == 'Scientific'])),
            random_state=42
        )
        training_df = pd.concat([policy_docs, scientific_docs])
        
        # Preprocess documents
        all_processed = df['text'].apply(self.preprocessor.preprocess)
        training_processed = training_df['text'].apply(self.preprocessor.preprocess)
        
        # Create and fit vectorizer
        self.vectorizer = self._create_vectorizer()
        training_dtm = self.vectorizer.fit_transform(training_processed)
        
        # Train LDA model
        self.lda_model = self._create_lda_model()
        self.lda_model.fit(training_dtm)
        
        # Transform full corpus
        full_dtm = self.vectorizer.transform(all_processed)
        topic_scores = self.lda_model.transform(full_dtm)
        
        return self._create_results(df, topic_scores, full_dtm)
    
    def _calculate_threat_relevance(self, text: str) -> float:
        """Calculate document's relevance to threats"""
        text = text.lower()
        relevance_score = 0
        for threat_seeds in THREAT_SEEDS.values():
            for seed in threat_seeds:
                relevance_score += text.count(seed)
        return relevance_score

class TopicPostprocessingApproach(TopicModelBase):
    """Post-process topics to align with threats"""
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Train basic model on full corpus
        all_processed = df['text'].apply(self.preprocessor.preprocess)
        
        self.vectorizer = self._create_vectorizer()
        dtm = self.vectorizer.fit_transform(all_processed)
        
        self.lda_model = self._create_lda_model()
        topic_scores = self.lda_model.fit_transform(dtm)
        
        # Get initial results
        scores_df, topics_df = self._create_results(df, topic_scores, dtm)
        
        # Post-process topics
        processed_topics = self._process_topics(topics_df)
        
        return scores_df, processed_topics
    
    def _process_topics(self, topic_words_df: pd.DataFrame) -> pd.DataFrame:
        """Align topics with threat categories"""
        # Calculate alignment scores
        alignment_scores = {}
        for topic_col in topic_words_df.columns:
            topic_words = topic_words_df[topic_col].tolist()
            alignments = {}
            for threat, seeds in THREAT_SEEDS.items():
                alignments[threat] = self._calculate_threat_alignment(topic_words, seeds)
            alignment_scores[topic_col] = alignments
        
        # Create alignment DataFrame
        alignment_df = pd.DataFrame(alignment_scores)
        
        # Assign threat categories
        topic_categories = {}
        for topic_col in topic_words_df.columns:
            best_threat = alignment_df[topic_col].idxmax()
            score = alignment_df[topic_col].max()
            topic_categories[topic_col] = best_threat if score > 0.1 else "other"
        
        # Add metadata to topics
        topic_words_df.loc["threat_category"] = pd.Series(topic_categories)
        topic_words_df.loc["alignment_score"] = [alignment_df[col].max() for col in topic_words_df.columns]
        
        return topic_words_df
    
    def _calculate_threat_alignment(self, topic_words: List[str], threat_seeds: List[str]) -> float:
        """Calculate alignment between topic words and threat seeds"""
        return len(set(topic_words) & set(threat_seeds)) / len(set(threat_seeds))

def run_all_models(df: pd.DataFrame, n_topics: int = 50):
    """Run all four topic modeling approaches and save results"""
    results = {}
    
    # 1. Guided LDA
    print("Running Guided LDA...")
    guided_lda = GuidedLDAApproach(n_topics=n_topics)
    guided_scores, guided_topics = guided_lda.fit_transform(df)
    results['guided_lda'] = (guided_scores, guided_topics)
    
    # 2. Domain Enhanced
    print("Running Domain Enhanced Approach...")
    domain_enhanced = DomainEnhancedApproach(n_topics=n_topics)
    domain_scores, domain_topics = domain_enhanced.fit_transform(df)
    results['domain_enhanced'] = (domain_scores, domain_topics)
    
    # 3. Document Filtering
    print("Running Document Filtering Approach...")
    doc_filtering = DocumentFilteringApproach(n_topics=n_topics)
    filtered_scores, filtered_topics = doc_filtering.fit_transform(df)
    results['doc_filtering'] = (filtered_scores, filtered_topics)
    
    # 4. Topic Postprocessing
    print("Running Topic Postprocessing Approach...")
    postprocessing = TopicPostprocessingApproach(n_topics=n_topics)
    postproc_scores, postproc_topics = postprocessing.fit_transform(df)
    results['postprocessing'] = (postproc_scores, postproc_topics)
    
    # Save all results
    for approach, (scores, topics) in results.items():
        scores.to_csv(f'Data/Output/{approach}_scores_100_full.csv', index=False)
        topics.to_csv(f'Data/Output/{approach}_topics_100_full.csv', index=False)
        
        # Calculate and save average coherence
        avg_coherence = topics.loc['coherence'].mean()
        print(f"\nAverage coherence for {approach}: {avg_coherence:.3f}")
        
        # If threat categories are available, calculate coverage
        if 'threat_category' in topics.index:
            categories = topics.loc['threat_category'].value_counts()
            print(f"\nThreat category coverage for {approach}:")
            print(categories)
    
    return results

def analyze_results(results: Dict):
    """Analyze and compare results from different approaches"""
    analysis = pd.DataFrame(columns=['Approach', 'Avg Coherence', 'Threat Coverage', 'Distinct Topics'])
    
    for approach, (scores, topics) in results.items():
        # Calculate metrics
        avg_coherence = topics.loc['coherence'].mean()
        
        # Calculate threat coverage if available
        threat_coverage = 0
        if 'threat_category' in topics.index:
            categories = topics.loc['threat_category'].value_counts()
            threat_coverage = len(categories[categories.index != 'other'])
        
        # Calculate topic distinctiveness
        distinct_topics = len(set(topics.iloc[:-2].values.flatten()))  # Exclude metadata rows
        
        # Add to analysis DataFrame
        analysis.loc[len(analysis)] = [approach, avg_coherence, threat_coverage, distinct_topics]
    
    return analysis

def visualize_topic_distribution(results: Dict):
    """
    Create visualizations of topic distributions
    Note: This requires matplotlib and seaborn
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    for approach, (scores, topics) in results.items():
        # Create topic distribution plot
        plt.figure(figsize=(12, 6))
        
        # Plot average topic scores
        avg_scores = scores.mean()
        sns.barplot(x=avg_scores.index, y=avg_scores.values)
        
        plt.title(f'Average Topic Scores - {approach}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'Data/Output/{approach}_topic_distribution.png')
        plt.close()
        
        # Create coherence plot
        plt.figure(figsize=(12, 6))
        coherence_scores = topics.loc['coherence']
        sns.barplot(x=coherence_scores.index, y=coherence_scores.values)
        
        plt.title(f'Topic Coherence Scores - {approach}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'Data/Output/{approach}_coherence_distribution.png')
        plt.close()

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('Data/Input/merged_Corpus_1961-2023.csv')  
    
    # Run all models
    print("\nRunning all topic modeling approaches...")
    results = run_all_models(df, n_topics=100)
    
    # Analyze results
    # print("\nAnalyzing results...")
    # analysis = analyze_results(results)
    # analysis.to_csv('Data/Output/model_comparison.csv', index=False)
    
    # # Create visualizations
    # print("\nCreating visualizations...")
    # visualize_topic_distribution(results)
    
    print("\nProcess complete! Results have been saved to the Output directory.")