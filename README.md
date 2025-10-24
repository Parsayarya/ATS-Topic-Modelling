# ATS-Topic-Modelling

This project demonstrates how to perform topic modeling using LDA with scikit-learn, how to compute coherence scores to evaluate the quality of discovered topics, and how to monitor the convergence of an LDA model across multiple training iterations or hyperparameter settings.

* TopicModelling.py:

Reads a CSV of raw text data, preprocesses the text (tokenization, stopwords removal, lemmatization, etc.), and fits an LDA model with user-defined hyperparameters.
Saves the resulting topic distributions and top words per topic.


* CoherenceScore.py:

Computes coherence scores for different LDA models (varying the number of topics).
Helps select the optimal number of topics by comparing coherence scores.


* ConvergenceTest.py:

Monitors LDA model convergence by tracking perplexity and log-likelihood over multiple partial fits (iterations).




* File Descriptions


1- TopicModelling.py


  * Key functions/classes:

preprocess_text(text: str) -> str: Cleans and tokenizes text (removing punctuation, special characters, and stopwords, then lemmatizing).
LDAV1(df, text_column, n_topics=30, n_top_words=25): Demonstrates how to run an LDA model using a specific set of hyperparameters.
LDAV2(df, text_column, n_topics=90, n_top_words=20): Shows an alternative set of hyperparameters (e.g., different prior settings, different max_iter, etc.).


  * What it does:

Loads a CSV file named merged_Corpus_1961-2023.csv from Data/Input/.
Preprocesses the text data.
Fits one or more LDA models to the document-term matrix using scikit-learn’s LatentDirichletAllocation.
Exports the topic loading scores and top topic words as CSV files into an output directory (e.g., Data/Output2/ or Data/Output/refined/).



2- CoherenceScore.py


  * Key functions/classes:

preprocess_text(text: str) -> str: Identical or similar text cleaner as in TopicModelling.
compute_coherence_score(topic_word_matrix, term_doc_matrix) -> float: Computes the topic coherence via word co-occurrence in documents.
evaluate_topic_models(df, text_column, min_topics=50, max_topics=100, step=10): Trains multiple LDA models (with different numbers of topics) and calculates their coherence scores.


  * What it does:

Reads the same merged_Corpus_1961-2023.csv.
For each n_topics between min_topics and max_topics, it trains an LDA model, computes a coherence score, and tracks the results.
Saves a plot of coherence scores vs. number of topics.
Saves the coherence scores for each topic setting in a CSV.


3- ConvergenceTest.py


  * Key functions/classes:

preprocess_text(text: str) -> str: Same as above, cleans textual data.
monitor_convergence(df, text_column, n_topics=90, learning_offset=50.0, max_iter=3): Trains an LDA model in partial fits, capturing perplexity and log-likelihood after each iteration to observe convergence behavior.

  * What it does:

Reads merged_Corpus_1961-2023.csv.
Performs partial fits of the LDA model for a given number of iterations (max_iter).
Logs perplexity and log-likelihood after each iteration.
Saves these metrics and plots them (perplexity vs. iteration and log-likelihood vs. iteration) to help see if the model converges over time.


