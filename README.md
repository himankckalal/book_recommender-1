## Content based book recommender system

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

books = pd.read_csv('books.csv', encoding = 'ISO-8859-1')
books.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>book_id</th>
      <th>best_book_id</th>
      <th>work_id</th>
      <th>books_count</th>
      <th>isbn13</th>
      <th>original_publication_year</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.00000</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>9.415000e+03</td>
      <td>9979.000000</td>
      <td>10000.000000</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5000.50000</td>
      <td>5.264697e+06</td>
      <td>5.471214e+06</td>
      <td>8.646183e+06</td>
      <td>75.712700</td>
      <td>9.754393e+12</td>
      <td>1981.987674</td>
      <td>4.002191</td>
      <td>5.400124e+04</td>
      <td>5.968732e+04</td>
      <td>2919.955300</td>
      <td>1345.040600</td>
      <td>3110.885000</td>
      <td>11475.893800</td>
      <td>1.996570e+04</td>
      <td>2.378981e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2886.89568</td>
      <td>7.575462e+06</td>
      <td>7.827330e+06</td>
      <td>1.175106e+07</td>
      <td>170.470728</td>
      <td>4.428246e+11</td>
      <td>152.576665</td>
      <td>0.254427</td>
      <td>1.573700e+05</td>
      <td>1.678038e+05</td>
      <td>6124.378132</td>
      <td>6635.626263</td>
      <td>9717.123578</td>
      <td>28546.449183</td>
      <td>5.144736e+04</td>
      <td>7.976889e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>8.700000e+01</td>
      <td>1.000000</td>
      <td>1.951703e+08</td>
      <td>-1750.000000</td>
      <td>2.470000</td>
      <td>2.716000e+03</td>
      <td>5.510000e+03</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>30.000000</td>
      <td>323.000000</td>
      <td>7.500000e+02</td>
      <td>7.540000e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2500.75000</td>
      <td>4.627575e+04</td>
      <td>4.791175e+04</td>
      <td>1.008841e+06</td>
      <td>23.000000</td>
      <td>9.780000e+12</td>
      <td>1990.000000</td>
      <td>3.850000</td>
      <td>1.356875e+04</td>
      <td>1.543875e+04</td>
      <td>694.000000</td>
      <td>196.000000</td>
      <td>656.000000</td>
      <td>3112.000000</td>
      <td>5.405750e+03</td>
      <td>5.334000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5000.50000</td>
      <td>3.949655e+05</td>
      <td>4.251235e+05</td>
      <td>2.719524e+06</td>
      <td>40.000000</td>
      <td>9.780000e+12</td>
      <td>2004.000000</td>
      <td>4.020000</td>
      <td>2.115550e+04</td>
      <td>2.383250e+04</td>
      <td>1402.000000</td>
      <td>391.000000</td>
      <td>1163.000000</td>
      <td>4894.000000</td>
      <td>8.269500e+03</td>
      <td>8.836000e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7500.25000</td>
      <td>9.382225e+06</td>
      <td>9.636112e+06</td>
      <td>1.451775e+07</td>
      <td>67.000000</td>
      <td>9.780000e+12</td>
      <td>2011.000000</td>
      <td>4.180000</td>
      <td>4.105350e+04</td>
      <td>4.591500e+04</td>
      <td>2744.250000</td>
      <td>885.000000</td>
      <td>2353.250000</td>
      <td>9287.000000</td>
      <td>1.602350e+04</td>
      <td>1.730450e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10000.00000</td>
      <td>3.328864e+07</td>
      <td>3.553423e+07</td>
      <td>5.639960e+07</td>
      <td>3455.000000</td>
      <td>9.790000e+12</td>
      <td>2017.000000</td>
      <td>4.820000</td>
      <td>4.780653e+06</td>
      <td>4.942365e+06</td>
      <td>155254.000000</td>
      <td>456191.000000</td>
      <td>436802.000000</td>
      <td>793319.000000</td>
      <td>1.481305e+06</td>
      <td>3.011543e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
books = books.loc[books['average_rating'] >= 3.5]

print(books.shape)
print(books.columns)
```

    (9661, 23)
    Index(['id', 'book_id', 'best_book_id', 'work_id', 'books_count', 'isbn',
           'isbn13', 'authors', 'original_publication_year', 'original_title',
           'title', 'language_code', 'average_rating', 'ratings_count',
           'work_ratings_count', 'work_text_reviews_count', 'ratings_1',
           'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 'image_url',
           'small_image_url'],
          dtype='object')
    


```python
#ratings = pd.read_csv('ratings.csv', encoding = 'ISO-8859-1')
#ratings.head()
```


```python
book_tags = pd.read_csv('book_tags.csv', encoding = 'ISO-8859-1')
book_tags.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30574</td>
      <td>167697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11305</td>
      <td>37174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>11557</td>
      <td>34173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8717</td>
      <td>12986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33114</td>
      <td>12716</td>
    </tr>
  </tbody>
</table>
</div>




```python
tags = pd.read_csv('tags.csv')
# drop the 'to-read', 'favorites' and 'currently-reading' tag
tags.drop([30574, 11557, 8717], inplace = True)
print(tags.tail())
```

           tag_id    tag_name
    34247   34247   Ｃhildrens
    34248   34248   Ｆａｖｏｒｉｔｅｓ
    34249   34249       Ｍａｎｇａ
    34250   34250      ＳＥＲＩＥＳ
    34251   34251  ｆａｖｏｕｒｉｔｅｓ
    


```python
book_tags_df = pd.merge(book_tags, tags, on = 'tag_id', how = 'inner').sort_values(by = 'count', ascending = False)

#row = [i for i, row in book_tags_df.iterrows() if row['tag_name'] == 'to-read']
#book_tags_df.drop(row, inplace = True)
book_tags_df.head()
book_tags_df.shape
```




    (970272, 4)




```python
#to_read = pd.read_csv('to_read.csv')
#to_read.head()
```


```python
# author based recommender

tfidf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3), min_df = 0, stop_words = 'english')
#tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(books['authors'])
tfidf_matrix.shape
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```


```python
#Construct a reverse map of indices and book titles
indices = pd.Series(books.index, index = books['title']).drop_duplicates()
```


```python
# Function that takes in book titles as input and outputs most similar books
def get_recommendations(title, cosine_sim = cosine_sim):
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return books['title'].iloc[book_indices]
```


```python
books['title'].head(20)
```




    0               The Hunger Games (The Hunger Games, #1)
    1     Harry Potter and the Sorcerer's Stone (Harry P...
    2                               Twilight (Twilight, #1)
    3                                 To Kill a Mockingbird
    4                                      The Great Gatsby
    5                                The Fault in Our Stars
    6                                            The Hobbit
    7                                The Catcher in the Rye
    8                 Angels & Demons  (Robert Langdon, #1)
    9                                   Pride and Prejudice
    10                                      The Kite Runner
    11                            Divergent (Divergent, #1)
    12                                                 1984
    13                                          Animal Farm
    14                            The Diary of a Young Girl
    15     The Girl with the Dragon Tattoo (Millennium, #1)
    16                 Catching Fire (The Hunger Games, #2)
    17    Harry Potter and the Prisoner of Azkaban (Harr...
    18    The Fellowship of the Ring (The Lord of the Ri...
    19                    Mockingjay (The Hunger Games, #3)
    Name: title, dtype: object




```python
# tag based recommender

books_with_tags = pd.merge(books, book_tags_df, left_on = 'book_id', right_on = 'goodreads_book_id', how = 'inner')
books_with_tags['tag_name'] = books_with_tags['tag_name'].astype('category')
#books_with_tags.info()
tfidf1 = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 3), min_df = 0, stop_words = 'english')
tfidf_matrix1 = tfidf1.fit_transform(books_with_tags['tag_name'].head(20000))
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
```


```python
# tag and author based recommender

temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>book_id</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>fantasy young-adult fiction harry-potter books...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>fantasy children children-s all-time-favorites...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>fantasy young-adult fiction harry-potter books...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>fantasy young-adult fiction harry-potter books...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>fantasy young-adult fiction harry-potter owned...</td>
    </tr>
  </tbody>
</table>
</div>




```python
books0 = pd.merge(books, temp_df, on = 'book_id', how = 'inner')
#books0.head()
```


```python
# function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''
```


```python
books0['authors'].apply(clean_data)
```




    0                                          suzannecollins
    1                               j.k.rowling,marygrandprì©
    2                                          stepheniemeyer
    3                                               harperlee
    4                                       f.scottfitzgerald
    5                                               johngreen
    6                                           j.r.r.tolkien
    7                                            j.d.salinger
    8                                                danbrown
    9                                              janeausten
    10                                         khaledhosseini
    11                                           veronicaroth
    12                   georgeorwell,erichfromm,celì¢lìïster
    13                                           georgeorwell
    14      annefrank,eleanorroosevelt,b.m.mooyaart-doubleday
    15                                stieglarsson,regkeeland
    16                                         suzannecollins
    17                    j.k.rowling,marygrandprì©,rufusbeck
    18                                          j.r.r.tolkien
    19                                         suzannecollins
    20                              j.k.rowling,marygrandprì©
    21                                            alicesebold
    22                              j.k.rowling,marygrandprì©
    23                              j.k.rowling,marygrandprì©
    24                              j.k.rowling,marygrandprì©
    25                                               danbrown
    26                              j.k.rowling,marygrandprì©
    27                                         williamgolding
    28                       williamshakespeare,robertjackson
    29                                           gillianflynn
                                  ...                        
    9631                                             amimckay
    9632                                           melodyanne
    9633                                      kurtvonnegutjr.
    9634                              johnm.gottman,nansilver
    9635                                            loislowry
    9636                                         davidgemmell
    9637                                        mariav.snyder
    9638                                            annetyler
    9639                              yukiomishima,johnnathan
    9640                                          wilbursmith
    9641                                          deeannegist
    9642                                        jefferydeaver
    9643                                     karenmariemoning
    9644                                            kazuekato
    9645                                     petermatthiessen
    9646                  steveperry,tomclancy,stevepieczenik
    9647                                      terriblackstock
    9648                                         irisjohansen
    9649                                            johnrawls
    9650                                          quinnloftis
    9651                                        oscarhijuelos
    9652                                              benokri
    9653                                         milescameron
    9654                                          ianmortimer
    9655                         michaelbuckley,peterferguson
    9656                                         ilonaandrews
    9657                                         roberta.caro
    9658                                       patricko'brian
    9659                                       peggyorenstein
    9660                                           johnkeegan
    Name: authors, Length: 9661, dtype: object




```python
def create_soup(x):
    return ''.join(x['authors']) + '' + ''.join(x['tag_name'])
```


```python
books0['soup'] = books0.apply(create_soup, axis = 1)
```


```python
# Use the CountVectorizer() instead of TF-IDF. 
# This is because one do not want to down-weight the presence of an author if he or she has written relatively more books.
# But will still fit_transform a tfidfvectorizer for comparison 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df = 0, stop_words = 'english')
count_matrix = cv.fit_transform(books0['soup'])
cosine_sim0 = cosine_similarity(count_matrix, count_matrix)
#books0.reset_index()
```


```python
tfidf2 = TfidfVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df = 0, stop_words = 'english')
tfidf_matrix2 = tfidf2.fit_transform(books0['soup'])
cosine_sim2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)
```


```python
print('author-based recommendations:')
print(get_recommendations('The Great Gatsby'), '\n')
print('tag-based recommendations:')
print(get_recommendations('The Great Gatsby', cosine_sim1), '\n')
print('author and tag based recommendations using CountVectorizer:')
print(get_recommendations('The Great Gatsby', cosine_sim0), '\n')
print('author and tag based recommendations using tf-idf:')
print(get_recommendations('The Great Gatsby', cosine_sim2))
```

    author-based recommendations:
    1183                                  Tender Is the Night
    2303                                This Side of Paradise
    3254                             The Beautiful and Damned
    3640                  The Curious Case of Benjamin Button
    7408                                    The Short Stories
    8683    The Billionaire's Obsession ~ Simon (The Billi...
    1279                                           The Aeneid
    922     The Alchemyst (The Secrets of the Immortal Nic...
    1633    The Magician (The Secrets of the Immortal Nich...
    2016    The Sorceress (The Secrets of the Immortal Nic...
    Name: title, dtype: object 
    
    tag-based recommendations:
    97         The Girl Who Played with Fire (Millennium, #2)
    197                                      The Color Purple
    590       The Absolutely True Diary of a Part-Time Indian
    813                          Tempted (House of Night, #6)
    1095    America (The Book): A Citizen's Guide to Democ...
    1204      Hard-Boiled Wonderland and the End of the World
    1293    god is Not Great: How Religion Poisons Everything
    1598                                           Night Road
    1699                            A Tale for the Time Being
    1799                                                 Ubik
    Name: title, dtype: object 
    
    author and tag based recommendations using CountVectorizer:
    7        The Catcher in the Rye
    31              Of Mice and Men
    27            Lord of the Flies
    3         To Kill a Mockingbird
    129     The Old Man and the Sea
    13                  Animal Farm
    1185        The Glass Menagerie
    130         The Grapes of Wrath
    781               The Awakening
    522     The Things They Carried
    Name: title, dtype: object 
    
    author and tag based recommendations using tf-idf:
    7                  The Catcher in the Rye
    31                        Of Mice and Men
    781                         The Awakening
    57     The Adventures of Huckleberry Finn
    130                   The Grapes of Wrath
    386                          The Crucible
    27                      Lord of the Flies
    649                     Uncle Tom's Cabin
    129               The Old Man and the Sea
    128       One Flew Over the Cuckoo's Nest
    Name: title, dtype: object
    

Author-based and tag-based recommenders have awful performance obviously. Although, when it comes to tag and author based recommender, the results of countvectorizer and tfidfvectorizer only have relatively small differences.


```python
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
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CountVectorizer</th>
      <th>TfidfVectorizer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lord of the Flies</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Great Gatsby</td>
      <td>The Great Gatsby</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1984</td>
      <td>Animal Farm / 1984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Of Mice and Men</td>
      <td>Keep the Aspidistra Flying</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brave New World</td>
      <td>Lord of the Flies</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Catcher in the Rye</td>
      <td>Cry, the Beloved Country</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fahrenheit 451</td>
      <td>Of Mice and Men</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Old Man and the Sea</td>
      <td>The Fall of the House of Usher</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Silas Marner</td>
      <td>A Modest Proposal</td>
    </tr>
    <tr>
      <th>9</th>
      <td>To Kill a Mockingbird</td>
      <td>A Modest Proposal and Other Satirical Works</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
