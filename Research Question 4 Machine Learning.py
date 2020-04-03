"""
CSE 163 Project, Authors: Karan Singh, Varun Venkatesh, Waiz Khan
This file creates a linear regression model to predict movies box office based
on features in the movie dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def select_features(df):
    """
    This method takes in a parameter of a dataframe and filters out features
    (columns) that have values that are too distinct or too correlated to each
    other.
    """
    revenue = df['gross']
    cor = df.corr()
    cor_target = abs(cor['gross'])
    relevant_features = cor_target[cor_target > 0.1]
    df = df.filter(list(relevant_features.index))
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                .astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    df = df.drop(df[to_drop], axis=1)
    df['gross'] = revenue
    return df


def fit_revenue(df):
    """
    This method takes in the parameter of a dataset and trains a linear
    regression model to predict gross revenue(box office) of a movies based on
    features such as budget, rating, user votes, etc.
    """
    df = pd.get_dummies(df)
    df = df.dropna()
    # filters out movies with zero budget to reduce model error
    df = df[df['budget'] > 0]
    X = df.drop(columns=['gross'])
    y = df['gross']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    print(X_train)
    # test cases
    # defining some predictions
    Xnew = [[50000000, 96, 8, 50000, 2004], [6000000, 120, 6, 100000, 2014]]
    # make a prediction
    ynew = model.predict(Xnew)
    # show the inputs and predicted outputs
    for i in range(len(Xnew)):
        print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
    # returns coefficient of determinant (r^2) value to measure accuracy
    return model.score(X, y)


def main():
    df = pd.read_csv("movies.csv", encoding='ISO-8859-1')
    df = select_features(df)
    print(fit_revenue(df))


if __name__ == "__main__":
    main()
