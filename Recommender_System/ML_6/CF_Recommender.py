import numpy as np
import pandas as pd


def read_data():
    """
        read the two csv files
    :return:
        movies: dataframe of movies' id , the movies' title and geners
        data:  dataframe of movies' id and the movies' title, ratings and user's id
    """
    rating = pd.read_csv('ContentBasedRecommenderSystem/ratings.csv')
    rating = rating[rating['userId']<201]
    movies = pd.read_csv('ContentBasedRecommenderSystem/movies.csv')
    movies = movies[movies['movieId']<201]
    data = pd.merge(rating, movies, on='movieId')
    data = data.drop(columns=['timestamp','genres'])
    return data, movies

def get_Matrix(data):
    """
    :param
        data: must contain userId , movieId and Rating as columns
    :return:
        moviemat: matrix userId as index , movieId as column ,rating as values
        moviemat.columns: all movies' id
        moviemat.index:  all user's id
    """
    moviemat = data.pivot_table(index ='userId',
                  columns ='movieId', values ='rating')
    moviemat = moviemat.replace(np.NaN,0)
    return moviemat, moviemat.columns, moviemat.index

def cosine_similarity_mat(mat):
    """

    :param
        mat: matrix userId as index , movieId as column ,rating as values
    :return:
        similarity: matrix movieId as index , movieId as column ,cosine similarity as values
    """
    numerator = mat.T @ mat
    norm = (mat * mat).sum(0, keepdims=True) ** .5
    similarity = numerator / norm / norm.T
    return similarity
def get_highest_similarity(array, n):
    """

    :param
        array: array of similarities
    :param
        n: no of similarities needed
    :return:
        array of nth highest similarity
    """
    return array.argsort()[-n:][::-1]

def get_highest_similar_to(movies_id_list,n):
    """

    :param movies_id_list: list of movies' id to get movies similar to
    :param n: no of movies needed
    :return: list of list of similar movies id and string of all movies names
    """
    data, movies = read_data()
    highest_similar_movies = [0]*len(movies_id_list)
    moviemat , movies_id, users_id  = get_Matrix(data)
    cosine_mat = cosine_similarity_mat(np.array(moviemat))
    for_printing = ""
    for count , element in enumerate(movies_id_list):
        indx = np.where(moviemat.columns== element)[0]
        if indx.size>0:
            highest_similar_movies[count] =movies_id[ get_highest_similarity(cosine_mat[indx[0]], n +1)]
            for_printing += f"{movies['title'][movies['movieId'] == element].values[0]} is similar to: \n"
            for counter, movie in enumerate(highest_similar_movies[count][1:]):
                for_printing += f"       {counter +1}-   {movies['title'][movies['movieId'] == movie].values[0]}\n"

    return highest_similar_movies, for_printing