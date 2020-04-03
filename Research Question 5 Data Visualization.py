import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
This file contains a method that plots every genre's total user votes from
1986 to 2019
"""


def highest_and_lowest_rated_genres_over_the_years(df):
    """
    This method takes in a dataframe of information about movies
    and uses it to plot every genre's total user votes from 1986 to 2019
    """
    genre_ratings_over_years = df.groupby(['year', 'genre'],
                                          as_index=False)['votes'].sum()
    print(genre_ratings_over_years)
    sns.relplot(x='year', y='votes', hue='genre', kind='line',
                            data=genre_ratings_over_years)
    plt.title("Highest and Lowest Viewer Votes per Genres over the Years")
    plt.xlabel("Years")
    plt.ylabel("Viewer Votes by Genre")
    plt.show()


def main():
    """
    This method reads in the dataset from the same directory
    and calls highest_and_lowest_rated_genres_over_the_years
    """
    df = pd.read_csv("movies.csv", encoding='ISO-8859-1')
    highest_and_lowest_rated_genres_over_the_years(df)


if __name__ == "__main__":
    main()
