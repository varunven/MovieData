"""
CSE 163 Project, Authors: Karan Singh, Varun Venkatesh, Waiz Khan
This file creates two data visualizations on a movie dataset.
The first visualization is a graph of Movie Revenue by Country
The second visualization is a graph of GDP by Country
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def highest_grossing_movies_and_country_gdp(df, gdf, gdp_df):
    """
    This method takes in a dataframe with information about movies,
    a geodataframe containing information about the shapes of countries,
    and a dataframe contaning information about all countries' GDPs
    This method creates the two plots, the first being Movie Revenue by Country
    The second is a plot of GDP by Country
    """
    g_min = df['gross'].min()
    g_max = df['gross'].max()
    fig, [ax1, ax2] = plt.subplots(1, ncols=2)
    gdf.plot(ax=ax1, color='#EEEEEE')
    gdf.plot(ax=ax2, color='#EEEEEE')
    merged = gdf.merge(df, left_on="name", right_on="country", how="inner")
    countries = merged.dissolve(by="name", aggfunc="sum")
    countries.plot(column="gross", legend=True, ax=ax1, vmin=g_min, vmax=g_max)
    ax1.set_title('Movie Revenue by Country')
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
    This method loads the datasets and calls
    highest_grossing_movies_and_country_gdp()
    """
    df = pd.read_csv("movies.csv", encoding='ISO-8859-1')
    df = df.drop(columns='released')
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df.rename(columns={'name': 'Title'}, inplace=True)
    df['country'] = df['country'].str\
        .replace('USA', 'United States of America')
    gdp_df = pd.read_csv("gdp_csv.csv")
    gdp_df = gdp_df.fillna(0)
    # We replace names of countries that don't align in both datasets
    # with ones that do
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('United States', 'United States of America')
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('Russian Federation', 'Russia')
    gdp_df['Country Name'] = gdp_df['Country Name'].str\
        .replace('Congo, Dem. Rep.', 'Dem. Rep. Congo')
    gdp_df = gdp_df.groupby("Country Name").mean()
    highest_grossing_movies_and_country_gdp(df, gdf, gdp_df)


if __name__ == "__main__":
    main()
