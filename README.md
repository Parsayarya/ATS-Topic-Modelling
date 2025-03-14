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


