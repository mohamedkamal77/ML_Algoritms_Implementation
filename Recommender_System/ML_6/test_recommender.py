from CF_Recommender import get_highest_similar_to, read_data

# get highest 10 similar to movies with id 1 and 4
highest,to_print =get_highest_similar_to([1,4], 10)
print(to_print)

# get 3 recommendation to user with id 200
data, movies = read_data()
movies_id = data['movieId'][data['userId']==200].iloc[data['rating'][data['userId']==200].values.argmax()]
highest,to_print =get_highest_similar_to([movies_id], 3)
print(to_print)