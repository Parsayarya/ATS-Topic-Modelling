import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df_papers = pd.read_csv('ATCM46_WithNames_Categories-WITH-TEXT.csv')
df_papers = df_papers[df_papers['type']=='WP']
df_measures = pd.read_csv('MeasureCorpus_Latest.csv')
df_measures.dropna(subset='Content',inplace=True)



# First, drop duplicates from df_papers based on Title
df_papers = df_papers.drop_duplicates(subset=['Title'])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Create TF-IDF matrices for both datasets
# Fill NaN values with empty string to handle missing text
papers_text = df_papers['text'].fillna('')
measures_text = df_measures['Content'].fillna('')

# Fit and transform both document sets
tfidf_matrix_papers = tfidf.fit_transform(papers_text)
tfidf_matrix_measures = tfidf.transform(measures_text)

# Calculate similarity between all papers and measures
similarities = cosine_similarity(tfidf_matrix_papers, tfidf_matrix_measures)

# Create empty lists to store the top 3 similar measures for each paper
similar_measures_1 = []
similar_measures_2 = []
similar_measures_3 = []
similar_measures_link_1 = []
similar_measures_link_2 = []
similar_measures_link_3 = []

for idx in range(len(df_papers)):
    # Get similarities for current paper
    paper_similarities = similarities[idx]
    
    # Get indices sorted by similarity (highest to lowest)
    sorted_indices = np.argsort(paper_similarities)[::-1]
    
    # Lists to store distinct measures
    distinct_titles = []
    distinct_doc_numbers = []
    used_titles = set()
    
    # Find top 3 distinct measures
    for measure_idx in sorted_indices:
        current_title = df_measures.iloc[measure_idx]['Subject']
        
        # Check if we've found a new distinct title
        if current_title not in used_titles:
            used_titles.add(current_title)
            distinct_titles.append(current_title)
            distinct_doc_numbers.append(df_measures.iloc[measure_idx]['Document_Number'])
            
            # Break if we've found 3 distinct measures
            if len(distinct_titles) == 3:
                break
    
    # Pad with empty strings if we found fewer than 3 distinct measures
    while len(distinct_titles) < 3:
        distinct_titles.append('')
        distinct_doc_numbers.append('')
    
    # Append titles and links
    similar_measures_1.append(distinct_titles[0])
    similar_measures_2.append(distinct_titles[1])
    similar_measures_3.append(distinct_titles[2])
    
    similar_measures_link_1.append(f'https://www.ats.aq/devAS/Meetings/Measure/{distinct_doc_numbers[0]}' if distinct_doc_numbers[0] != '' else '')
    similar_measures_link_2.append(f'https://www.ats.aq/devAS/Meetings/Measure/{distinct_doc_numbers[1]}' if distinct_doc_numbers[1] != '' else '')
    similar_measures_link_3.append(f'https://www.ats.aq/devAS/Meetings/Measure/{distinct_doc_numbers[2]}' if distinct_doc_numbers[2] != '' else '')

# Create final dataframe
result_df = pd.DataFrame({
    'Generated_Name': df_papers['Generated_Name'],
    'Title': df_papers['Title'],
    'Category': df_papers['Category'],
    'Link': df_papers['link'],
    'most_similar_measure_1': similar_measures_1,
    'most_similar_measure_2': similar_measures_2,
    'most_similar_measure_3': similar_measures_3,
    'Link_1': similar_measures_link_1,
    'Link_2': similar_measures_link_2,
    'Link_3': similar_measures_link_3
})

result_df.to_csv('ATCM46_WP_TextCorrelation_WithMeasures_Dim_withLinks_Distincts.csv', index = False)