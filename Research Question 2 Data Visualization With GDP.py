"""
This file contains a method that plots every country based on the ratio between
budgets and critics'ratings on one plot
It also plots every country based on its GDP
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def highest_budget_to_rating_and_country_gdp(df, gdf, gdp_df):
    """
    This method takes in a dataframe of information about movies,
    a geopandas dataframe that contains information about world countries'
    shapes, and a dataframe that contains information about the gdp of every
    country.
    It plots a plot of every country based on the ratio between budgets and
    critics' ratings and a plot for every country based on its GDP.
    """
    df = df[df['budget'] > 0]
    df['budget_to_score'] = df['score'] / df['budget']
    b_min = df['budget_to_score'].min()
    b_max = df['budget_to_score'].max()
    merged = gdf.merge(df, left_on="name", right_on="country", how="inner")
    fig, [ax1, ax2] = plt.subplots(1, ncols=2)
    gdf.plot(ax=ax1, color='#EEEEEE')
    gdf.plot(ax=ax2, color='#EEEEEE')
    countries = merged.dissolve(by="name", aggfunc="sum")
    countries.plot(column="budget_to_score", legend=True, ax=ax1, vmin=b_min,
                   vmax=b_max)
    ax1.set_title('Rating to Budget Ratio by Country')
    merged_gdp = gdf.merge(gdp_df, left_on="name", right_on="Country Name",
                           how="inner")
    gdp_min = merged_gdp['Value'].min()
    gdp_max = merged_gdp['Value'].max()
    merged_gdp.plot(column="Value", legend=True, ax=ax2, vmin=gdp_min,
                    vmax=gdp_max)
    ax2.set_title('GDP by Country')
    plt.show()


def main():
    """
    This is the main method that loads all datasets from the same directory
    and the geopandas library
    It realigns information in one dataframe to match the names of countries in
    another
    It runs the highest_budget_to_rating_and_country_gdp method
    """
    df = pd.read_csv("movies.csv", encoding='ISO-8859-1')
    df = df.drop(columns='released')
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df.rename(columns={'name': 'Title'}, inplace=True)
    df['country'] = df['country'].str\
        .replace('USA', 'United States of America')
    gdp_df = pd.read_csv("gdp_csv.csv")
    gdp_df = gdp_df.fillna(0)
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('United States', 'United States of America')
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('Russian Federation', 'Russia')
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('Congo, Dem. Rep.', 'Dem. Rep. Congo')
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('Venezuela, RB', 'Venezuela')
    gdp_df = gdp_df.groupby("Country Name").mean()
    highest_budget_to_rating_and_country_gdp(df, gdf, gdp_df)


if __name__ == "__main__":
    main()
