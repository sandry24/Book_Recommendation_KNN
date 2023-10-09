import pandas as pd
from sklearn.neighbors import NearestNeighbors

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

print(df_books.head())
print(df_ratings.head())

# data cleaning
df = df_ratings

counts1 = df['user'].value_counts()
counts2 = df['isbn'].value_counts()

# remove users with less than 200 reviews and books with less than 100 reviews
df = df[~df['user'].isin(counts1[counts1 < 200].index)]
df = df[~df['isbn'].isin(counts2[counts2 < 100].index)]

# merge both dataframes
df = pd.merge(right=df, left=df_books, on="isbn")

df = df.drop_duplicates(["title", "user"])

# pivot the dataframe, make a grid [title, user] = rating
piv = df.pivot(index='title', columns='user', values='rating').fillna(0)

print(piv.head())

# create the model
matrix = piv.values
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(matrix)


# function to return recommended books - this will be tested
def get_recommends(book=""):
    X = piv.loc[book].array.reshape(1, -1)
    distances, indices = model_knn.kneighbors(X, n_neighbors=6)
    recommended_books = []

    for distance, index in zip(distances[0], indices[0]):
        if distance != 0:
            book_title = piv.index[index]
            recommended_books.append([book_title, distance])

    # reverse the list to get the recommendation from most similar to least
    recommended_books = [book, recommended_books[::-1]]
    return recommended_books


# test the model
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)


def book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
    for i in range(2):
        if recommends[1][i][0] not in recommended_books:
            test_pass = False
        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
            test_pass = False
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")


book_recommendation()
