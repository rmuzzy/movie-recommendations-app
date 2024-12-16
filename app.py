import dash
from dash import html, Input, Output, State, ctx, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os

movies = pd.read_csv(
    "https://liangfgithub.github.io/MovieData/movies.dat?raw=true", sep="::", engine="python", header=None, encoding="ISO-8859-1"
)
movies.columns = ["MovieID", "Title", "Genres"]
movies["MovieID"] = movies["MovieID"].astype(str)
sample_movies = movies.head(100)
R = pd.read_csv("https://d3c33hcgiwev3.cloudfront.net/I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv?Expires=1734480000&Signature=POU53r-wt9D3qAj9LesIXs7WFzJUJyfoon7QgMqkNgXHE8rfoFoW0BGX0NwfTPp2EOhtv1BG2Ew0YRDHu2T4I5TKI2q8W-1Hn1NlNjCMr8hBWEt-cXn8PUDa-HmjkW-nPvTjDgHL2GPTRkLlMRT7-FuN1Nr2WFHW7I6IekMEsDE_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A", index_col=0)
S = pd.read_csv("./similarity_matrix_top30.csv", index_col=0)
popular_movies_saved = pd.read_csv("./popular_movies.csv")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])


def create_star_rating(movie_id):
    return html.Div(
        id=f"rating-{movie_id}",
        children=[
            html.Span(
                "\u2605",  # Unicode star character
                id=f"star-{movie_id}-{i}",
                className="star-rating",
                n_clicks=0,
            )
            for i in range(1, 6)
        ],
        className="d-flex justify-content-center align-items-center",
    )

def create_movie_card(movie):
    return dbc.Col(
        html.Div(
            [
                html.Img(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie['MovieID']}.jpg",
                    alt=f"{movie['Title']} Poster",
                    className="movie-poster mb-2",
                ),
                html.H6(
                    movie["Title"],
                    id=f"title-tooltip-{movie['MovieID']}",
                    className="movie-title mt-2 text-center",
                ),
                dbc.Tooltip(
                    movie["Title"],
                    target=f"title-tooltip-{movie['MovieID']}",
                    placement="top",
                ),
                create_star_rating(movie["MovieID"]),
            ],
            className="text-center",
        ),
        xs=12,
        sm=6,
        md=4,
        lg=3,
        xl=2,
        className="mb-4",
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Movie Recommendation System",
                    className="text-center mt-4 mb-4 text-white",
                ),
                html.P(
                    "Rate the movies below by clicking the stars.",
                    className="text-center mb-4 text-white",
                ),
            ],
            className="header-section",
        ),
        html.Div(
            [
                dcc.Store(
                    id="ratings-store",
                    data={movie_id: 0 for movie_id in sample_movies["MovieID"]},
                ),
                html.Div(
                    dbc.Row(
                        [
                            create_movie_card(movie)
                            for _, movie in sample_movies.iterrows()
                        ]
                    ),
                    className="movie-rating-container",
                ),
            ],
            className="p-4 movie-rating-section",
        ),
        html.Div(
            [
                dbc.Button(
                    "Get Recommendations",
                    id="submit-button",
                    n_clicks=0,
                    color="primary",
                    className="mt-4",
                ),
            ],
            className="button-section",
        ),
        html.Div(
            id="recommendations-output",
            className="recommendations-section",
        ),
    ]
)


@app.callback(
    [
        Output(f"star-{movie_id}-{i}", "style")
        for movie_id in sample_movies["MovieID"]
        for i in range(1, 6)
    ]
    + [Output("ratings-store", "data")],
    [
        Input(f"star-{movie_id}-{i}", "n_clicks")
        for movie_id in sample_movies["MovieID"]
        for i in range(1, 6)
    ],
    [State("ratings-store", "data")],
)
def update_star_colors(*args):
    ratings_store = args[-1]

    if ratings_store is None:
        ratings_store = {movie_id: 0 for movie_id in sample_movies["MovieID"]}

    if ctx.triggered_id:
        try:
            _, clicked_movie_id, clicked_star = ctx.triggered_id.split("-")
            clicked_movie_id = str(clicked_movie_id)
            clicked_star = int(clicked_star)
            ratings_store[clicked_movie_id] = clicked_star
        except ValueError:
            pass

    updated_styles = []
    for movie_id in sample_movies["MovieID"]:
        for i in range(1, 6):
            color = "gold" if i <= ratings_store.get(movie_id, 0) else "gray"
            updated_styles.append(
                {"cursor": "pointer", "fontSize": "20px", "color": color}
            )

    return updated_styles + [ratings_store]


@app.callback(
    Output("recommendations-output", "children"),
    Input("submit-button", "n_clicks"),
    State("ratings-store", "data"),
)
def get_recommendations(n_clicks, ratings_store):
    if n_clicks > 0:
        new_user = pd.Series(dtype=float)
        for movie_id, rating in ratings_store.items():
            if rating > 0:
                new_user[f"m{movie_id}"] = rating

        new_user = new_user.reindex(R.columns, fill_value=np.nan)
        recommendations = myIBCF(new_user, S, R, popular_movies_saved)

        recommendations_html = html.Div(
            [
                html.H3("Top Recommended Movies:", className="text-center my-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardImg(
                                        src=f"https://liangfgithub.github.io/MovieImages/{movie_id.lstrip('m')}.jpg",
                                        top=True,
                                        className="recommendation-card-img",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H5(
                                                f"#{rank+1}: {movies.loc[movies['MovieID'] == movie_id.lstrip('m'), 'Title'].values[0]}",
                                                className="card-title text-center recommendation-card-title",
                                            ),
                                        ]
                                    ),
                                ],
                                className="recommendation-card",
                            ),
                            xs=12,
                            sm=6,
                            md=4,
                            lg=3,
                            xl=3,
                        )
                        for rank, movie_id in enumerate(
                            recommendations
                        )
                    ],
                    className="justify-content-center",
                ),
            ]
        )

        return recommendations_html
    return html.Div("Rate movies and click the button to get recommendations.")


def myIBCF(newuser, S, R, popular_movies_saved):
    newuser = pd.Series(newuser, index=R.columns)
    predictions = pd.Series(index=R.columns, dtype=float)

    for i in R.columns:
        if pd.notna(newuser[i]):
            continue

        neighbors = S.loc[i].dropna()
        rated_neighbors = neighbors.index.intersection(newuser.dropna().index)

        if len(rated_neighbors) == 0:
            predictions[i] = np.nan
            continue

        numerator = (neighbors[rated_neighbors] * newuser[rated_neighbors]).sum()
        denominator = neighbors[rated_neighbors].sum()

        predictions[i] = numerator / denominator if denominator != 0 else np.nan

    top_10_predictions = predictions.nlargest(10).dropna()
    top_10_recommendations = top_10_predictions.index.tolist()

    if len(top_10_recommendations) < 10:
        rated_movies = set(newuser.dropna().index)
        fallback_movies = popular_movies_saved.loc[
            ~popular_movies_saved["MovieID"].isin(rated_movies), "MovieID"
        ]
        top_10_recommendations.extend(
            fallback_movies.head(10 - len(top_10_recommendations)).tolist()
        )

    return top_10_recommendations


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
