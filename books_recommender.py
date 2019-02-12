import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

books = pd.read_csv('books.csv', encoding = 'ISO-8859-1')
print(books.describe())

books = books.loc[books['average_rating'] >= 3.5]

print(books.shape)
print(books.columns)


#ratings = pd.read_csv('ratings.csv', encoding = 'ISO-8859-1')
#print(ratings.head())


book_tags = pd.read_csv('book_tags.csv', encoding = 'ISO-8859-1')
print(book_tags.head())


tags = pd.read_csv('tags.csv')
# drop the 'to-read', 'favorites' and 'currently-reading' tag
tags.drop([30574, 11557, 8717], inplace = True)
print(tags.tail())


book_tags_df = pd.merge(book_tags, tags, on = 'tag_id', how = 'inner').sort_values(by = 'count', ascending = False)

#row = [i for i, row in book_tags_df.iterrows() if row['tag_name'] == 'to-read']
#book_tags_df.drop(row, inplace = True)
print(book_tags_df.head())
print(book_tags_df.shape)


#to_read = pd.read_csv('to_read.csv')
#print(to_read.head())


# author based recommender

tfidf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3), min_df = 0, stop_words = 'english')
#tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(books['authors'])
print(tfidf_matrix.shape)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#Construct a reverse map of indices and book titles
indices = pd.Series(books.index, index = books['title']).drop_duplicates()


# Function that takes in book titles as input and outputs most similar books
def get_recommendations(title, cosine_sim = cosine_sim):
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return books['title'].iloc[book_indices]


books['title'].head(20)


# tag based recommender

books_with_tags = pd.merge(books, book_tags_df, left_on = 'book_id', right_on = 'goodreads_book_id', how = 'inner')
books_with_tags['tag_name'] = books_with_tags['tag_name'].astype('category')
#books_with_tags.info()
tfidf1 = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3), min_df = 0, stop_words = 'english')
tfidf_matrix1 = tfidf1.fit_transform(books_with_tags['tag_name'].head(20000))
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)


# tag and author based recommender

temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head()


books0 = pd.merge(books, temp_df, on = 'book_id', how = 'inner')
#books0.head()


# function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''


books0['authors'].apply(clean_data)


def create_soup(x):
    return ''.join(x['authors']) + '' + ''.join(x['tag_name'])


books0['soup'] = books0.apply(create_soup, axis = 1)



# Use the CountVectorizer() instead of TF-IDF. 
# This is because you do not want to down-weight the presence of an author if he or she has written relatively more books.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df = 0, stop_words = 'english')
count_matrix = cv.fit_transform(books0['soup'])
cosine_sim0 = cosine_similarity(count_matrix, count_matrix)
#books0.reset_index()


tfidf2 = TfidfVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df = 0, stop_words = 'english')
tfidf_matrix2 = tfidf2.fit_transform(books0['soup'])
cosine_sim2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)



#print('author-based recommendations:')
#print(get_recommendations('Brave New World'), '\n')
#print('tag-based recommendations:')
#print(get_recommendations('Brave New World', cosine_sim1), '\n')
#print('author and tag based recommendations using CountVectorizer:')
#print(get_recommendations('Brave New World', cosine_sim0), '\n')
#print('author and tag based recommendations using TfidfVectorizer:')
#print(get_recommendations('Brave New World', cosine_sim2))

dict = {'CountVectorizer': [x for i,x in enumerate(get_recommendations('Animal Farm', cosine_sim0))], 
        'TfidfVectorizer': [x for i,x in enumerate(get_recommendations('Animal Farm', cosine_sim2))]}
results = pd.DataFrame(data = dict)
results

