import pandas as pd
import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
from dash import dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_cytoscape as cyto
import numpy as np

# process data
# datalist = pd.read_excel('IMDb movies.xlsx')
datalist = pd.read_csv("IMDb movies.csv", low_memory=False)
datalist2 = pd.read_csv("IMDb ratings.csv", low_memory=False)
df = datalist.merge(datalist2, on='imdb_title_id')
row_index = df[(df['year'].str.contains(r'TV Movie 2019', na=True))].index
df.drop(row_index, inplace=True)
df['year'] = df['year'].astype(int)
df.drop_duplicates(inplace=True)
gen_dict = {}
for genre in df['genre']:
    if len(genre.split(", ")) > 1:
        for item in genre.split(", "):
            gen_dict[item] = gen_dict.get(item, 0) + 1
    elif len(genre.split(", ")) == 1:
        gen_dict[genre] = gen_dict.get(genre, 0) + 1
company_dict = {}
dfm = df['production_company'].value_counts().reset_index()[:10]
dfm.rename(columns={'index': "company", 'production_company': 'count'}, inplace=True)
for company in dfm['company']:
    company_dict[company] = company_dict.get(company, 0) + 1


def getgenre(data, genre):
    if genre == "All" or genre == None:
        dfm = data
    else:
        dfm = data[data['genre'].str.contains(genre, na=False)]
    return dfm


def director(data):
    data['director'].fillna('Unknown', inplace=True)
    data['director'] = data["director"].apply(lambda x: x.split(", ")[0])
    dataset1 = data[['director', 'avg_vote']].groupby('director').mean().reset_index()
    dataset2 = data[['director', 'avg_vote']].groupby('director').count().reset_index()
    dataset2.rename(columns={'avg_vote': "film counts"}, inplace=True)
    dataset = dataset1.merge(dataset2, on='director')
    return dataset


def topselection(data, sortvalue, num):
    return data.sort_values(by=sortvalue, ascending=False).head(num)


def getyear(data, time):
    return data[(data['year'] >= time[0]) & (data['year'] <= time[1])]


def topdirector(data, genre, time, col):
    dfm = getyear(data, time)
    dfm = topselection(director(getgenre(dfm, genre)), col, 10)
    dfm['avg_vote'] = dfm['avg_vote'].round(decimals=2)
    return dfm


def genre_year(data, name):
    genres = pd.DataFrame(index=[], columns=['year', "name", "count", "full_rating", "max_rating", 'min_rating'])
    for index, movie in data.iterrows():
        for genre in movie["genre"].split(", "):
            if (len(genres[(genres["name"] == genre) & (genres['year'] == movie['year'])]) < 1):
                genres.loc[len(genres.index)] = [movie['year'], genre, 0, 0, movie['avg_vote'], movie['avg_vote']]
            genres.loc[(genres["name"] == genre) & (genres['year'] == movie['year']), "count"] += 1
            genres.loc[(genres["name"] == genre) & (genres['year'] == movie['year']), "full_rating"] += movie[
                'avg_vote']
            if (len(genres[(genres['name'] == genre) & (genres['year'] == movie['year']) & (
                    genres['max_rating'] > movie['avg_vote'])]) < 1):
                genres.loc[(genres["name"] == genre) & (genres['year'] == movie['year']), "max_rating"] = movie[
                    'avg_vote']
            if (len(genres[(genres['name'] == genre) & (genres['year'] == movie['year']) & (
                    genres['min_rating'] < movie['avg_vote'])]) < 1):
                genres.loc[(genres["name"] == genre) & (genres['year'] == movie['year']), "min_rating"] = movie[
                    'avg_vote']
    genres['avg_rating'] = genres['full_rating'] / genres['count']
    genres = genres[genres['name'] == name]
    return genres


def genre_backup(data, genre):
    data['genre'].fillna('Unknown', inplace=True)
    data['genre'] = data["genre"].apply(lambda x: x.split(", ")[0])
    dataset1 = data[['year', 'genre', 'avg_vote']].groupby(['year', 'genre']).mean().reset_index()
    dataset2 = data[['year', 'genre', 'avg_vote']].groupby(['year', 'genre']).max().reset_index()
    dataset3 = data[['year', 'genre', 'avg_vote']].groupby(['year', 'genre']).min().reset_index()
    dataset2.rename(columns={'avg_vote': "max rating"}, inplace=True)
    dataset3.rename(columns={'avg_vote': "min rating"}, inplace=True)
    dataset = dataset1.merge(dataset2, on=['genre', 'year'])
    dataset = dataset.merge(dataset3, on=['genre', 'year'])
    dataset = dataset[dataset['genre'] == genre]
    return dataset


def age(data, age, name):
    if age == 'Under 18':
        dataset = data[['year', 'genre', 'males_0age_avg_vote', 'females_0age_avg_vote']]
        dataset.rename(columns={'males_0age_avg_vote': 'male_avg_vote', 'females_0age_avg_vote': 'female_avg_vote'},
                       inplace=True)
    elif age == '18-30':
        dataset = data[['year', 'genre', 'males_18age_avg_vote', 'females_18age_avg_vote']]
        dataset.rename(columns={'males_18age_avg_vote': 'male_avg_vote', 'females_18age_avg_vote': 'female_avg_vote'},
                       inplace=True)
    elif age == '30-45':
        dataset = data[['year', 'genre', 'males_30age_avg_vote', 'females_30age_avg_vote']]
        dataset.rename(columns={'males_30age_avg_vote': 'male_avg_vote', 'females_30age_avg_vote': 'female_avg_vote'},
                       inplace=True)
    elif age == '45+':
        dataset = data[['year', 'genre', 'males_45age_avg_vote', 'females_45age_avg_vote']]
        dataset.rename(columns={'males_45age_avg_vote': 'male_avg_vote', 'females_45age_avg_vote': 'female_avg_vote'},
                       inplace=True)
    elif age is None:
        dataset = data[['year', 'genre', 'males_allages_avg_vote', 'females_allages_avg_vote']]
        dataset.rename(
            columns={'males_allages_avg_vote': 'male_avg_vote', 'females_allages_avg_vote': 'female_avg_vote'},
            inplace=True)
    elif age == 'All':
        dataset = data[['year', 'genre', 'males_allages_avg_vote', 'females_allages_avg_vote']]
        dataset.rename(
            columns={'males_allages_avg_vote': 'male_avg_vote', 'females_allages_avg_vote': 'female_avg_vote'},
            inplace=True)
    dataset['genre'] = dataset["genre"].apply(lambda x: x.split(", ")[0])
    if name == "All" or name is None:
        dataset1 = dataset[['year', 'male_avg_vote', 'female_avg_vote']].groupby(
            ['year']).mean().reset_index()
        return dataset1
    else:
        dataset1 = dataset[['year', 'genre', 'male_avg_vote', 'female_avg_vote']].groupby(
            ['year', 'genre']).mean().reset_index()
        dataset = dataset1[dataset1['genre'] == name]
        return dataset


def country_ranking(data):
    data['worlwide_gross_income'] = data['worlwide_gross_income'].replace('[$INRKGBP ]', '', regex=True).astype(float)
    data['country'].fillna("Unknown", inplace=True)
    data['country'] = data['country'].apply(lambda x: x.split(", ")[0])
    dataset = data['country'].value_counts().reset_index()[:10]
    dataset.rename(columns={'index': "country", 'country': 'count'}, inplace=True)
    dataset = dataset.sort_values(by='count', ascending=True)
    return dataset


def income(data):
    data['worlwide_gross_income'] = data['worlwide_gross_income'].replace('[$INRKGBP ]', '', regex=True).astype(float)
    data['title'].fillna("Unknown", inplace=True)
    dataset = data[['year', 'title', 'worlwide_gross_income', 'avg_vote']].groupby(
        ['year', 'title']).mean().reset_index()
    dataset = dataset.sort_values(by='worlwide_gross_income', ascending=False)[:10]
    data = dataset.sort_values(by='worlwide_gross_income', ascending=True)
    return data


def company(data):
    dfm = data['production_company'].value_counts().reset_index()[:10]
    dfm.rename(columns={'index': "company", 'production_company': 'count'}, inplace=True)
    dfm = dfm.sort_values(by='count', ascending=True)
    return dfm


def duration(data, company):
    if company == "All" or company == None:
        dfm = data[['year', 'duration']]
    else:
        dfm = data[data['production_company'] == company]
    return dfm[['year', 'duration', 'genre']]


def relationship(data, num, name):
    data = getgenre(data, name)
    actor_film_records = data['actors'].dropna().values
    actors_relationship = dict()
    for r in actor_film_records:
        actors = set()
        curr_list = [x.strip() for x in r.split(',')]
        for cr in curr_list:
            actors.add(cr)
        actors = list(actors)
        for index, i in enumerate(actors):
            for j in actors[index + 1:]:
                if actors_relationship.get((i, j)) is None:
                    actors_relationship[(i, j)] = 0
                actors_relationship[(i, j)] += 1
    for k in list(actors_relationship.keys()):
        if actors_relationship[k] >= num:
            continue
        else:
            del actors_relationship[k]
    return actors_relationship


def high_value(data):
    dataset = data[data['avg_vote'] >= 8]
    return dataset


# layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(
        className="header",
        children=[
            html.P('Movie GO GO GO', className="title")
        ]
    ),
    html.Div([
        html.Div(
            [
                html.Div([
                    html.P('Top 10 Director', className="table_title"),
                    html.Div([html.Div([
                        html.P('Filter By genre:', className="filter_genre"),
                        dcc.Dropdown(
                            options=[{'label': k, 'value': k} for k in gen_dict.keys()],
                            value='All',
                            id='genre'
                        ),
                        html.P('Order By: ', className="order"),
                        dcc.RadioItems(
                            options=[
                                {'label': 'ratings', 'value': 'avg_vote'},
                                {'label': 'film counts', 'value': 'film counts'},
                            ],
                            value='avg_vote',
                            labelStyle={'display': 'inline-block'},
                            id='radio'
                        ),
                        html.P('Year Range: ', className="year_range"),
                        dcc.RangeSlider(
                            id='my-range-slider',
                            min=1894,
                            max=2021,
                            step=1,
                            value=[1894, 2021],
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className='banner_left'),
                        html.Div(id='Director', children=[], className='Table')
                    ], className='center'),
                ], className="left"),
                html.Div([
                    html.P('Ratings of different films from 1920 to 2020', className="second_title"),
                    html.Div([
                        html.P('genre: ', className="title_genre"),
                        dcc.Dropdown(
                            options=[{'label': k, 'value': k} for k in gen_dict.keys()],
                            value='Crime',
                            id='genre2',
                            clearable=False,
                        )
                    ], className="chart_condition"),
                    dcc.Graph(id="line-chart", figure={}),
                ], className="right")
            ], className="container"
        ),
        html.Div([
            html.Div([
                html.P('Difference of voting preference between Male and Female', className="third_title"),
                html.Div([
                    html.Div([
                        html.P('Age group: ', className="age_group"),
                        dcc.Dropdown(
                            options=[{'label': '0 - 18', 'value': 'Under 18'},
                                     {'label': '18 - 30', 'value': '18-30'},
                                     {'label': '30 - 45', 'value': '30-45'},
                                     {'label': '> 45', 'value': '45+'}
                                     ],
                            value='All',
                            id='age'
                        ),
                        html.P('Filter by genre: ', className="genre_difference"),
                        dcc.Dropdown(
                            options=[{'label': k, 'value': k} for k in gen_dict.keys()],
                            value='All',
                            id='genre-filter'
                        )
                    ], className="under_left"),
                    html.Div([
                        dcc.Graph(id="dif-chart", figure={}),
                    ], className="third_chart")
                ], className="center2")
            ], className="under_component")
        ], className="second_part"),
        html.Div([
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label='Top 10 Countries and Top 10 Movies', children=[
                        html.P('Year range: ', className="top_year"),
                        dcc.RangeSlider(
                            id='Tab_slider',
                            min=1920,
                            max=2020,
                            step=1,
                            value=[1920, 2020],
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        dcc.Graph(id="subplot",
                                  figure={}
                                  )
                    ], className="Country"),
                    dcc.Tab(label='Company Overview', children=[
                        html.Div([
                            html.P('Company: ', className="map_company"),
                            dcc.Dropdown(
                                options=[{'label': k, 'value': k} for k in company_dict.keys()],
                                value='Columbia Pictures',
                                id='map_choose_company',
                                clearable=False,
                            ),
                            html.P('Genre: ', className="map_genre"),
                            dcc.Dropdown(
                                options=[{'label': k, 'value': k} for k in gen_dict.keys()],
                                value='All',
                                id='map_choose_genre'
                            )
                        ], className="map_condition"),
                        dcc.Graph(id="country_map",
                                  figure={}
                                  )
                    ], className="map"),
                    dcc.Tab(label='Actor network graph', children=[
                        html.Div([
                            html.P('Genre: ', className="graph_genre"),
                            dcc.Dropdown(
                                options=[{'label': k, 'value': k} for k in gen_dict.keys()],
                                value='All',
                                id='graph_choose_genre',

                            ),
                            html.P('Cooperation level: ', className="graph_correlation"),
                            dcc.Dropdown(
                                options=[
                                    {'label': '>=4', 'value': 4},
                                    {'label': '>=5', 'value': 5},
                                    {'label': '>=6', 'value': 6},
                                    {'label': '>=7', 'value': 7},
                                    {'label': '>=8', 'value': 8},
                                    {'label': '>=9', 'value': 9},
                                    {'label': '>=10', 'value': 10},
                                    {'label': '>=11', 'value': 11}
                                ],
                                value=6,
                                clearable=False,
                                id='graph_choose_correlation'
                            )
                        ], className="graph_condition"),
                        html.Div(
                            children=[
                                cyto.Cytoscape(
                                    id='cytoscape',
                                    elements=[],
                                    layout={'name': 'random'},
                                    style={'width': '100%', 'height': '408px'},
                                    stylesheet=[
                                        {
                                            'selector': 'node',
                                            'style': {
                                                'content': 'data(label)',
                                                'width': 20,
                                                'height': 20
                                            }
                                        },
                                        {
                                            'selector': '.red',
                                            'style': {
                                                'background-color': 'red',
                                                'line-color': 'blue',
                                            }
                                        },
                                        {
                                            'selector': '[weight <= 5]',
                                            'style': {
                                                'line-color': '#767676',
                                                'line-style': 'dashed',
                                                'line-width': 1,
                                            }
                                        },
                                        {
                                            'selector': '[weight = 6]',
                                            'style': {
                                                'line-color': '#B1B1B1',
                                                'line-width': 0.8,
                                                'line-opacity': 0.3
                                            }
                                        },
                                        {
                                            'selector': '[weight >= 7]',
                                            'style': {
                                                'background-color': 'blue',
                                                'line-color': 'black',
                                                'line-width': 3,
                                                'line-dash-offset': 24
                                            }
                                        }

                                    ],
                                    zoomingEnabled=False,
                                    zoom=0.7
                                )
                            ], className='graph_area'
                        )
                    ]
                            )
                ], className='custom-tabs-container', parent_className='custom-tabs')
            ], className="Tab_component")
        ], className="third_part"),
    ], className="content")
])


@app.callback(
    Output(component_id='Director', component_property='children'),
    Input(component_id='genre', component_property='value'),
    Input(component_id='my-range-slider', component_property='value'),
    Input(component_id='radio', component_property='value')
)
def update_table(input_value, year, col):
    dfm = topdirector(df, input_value, year, col)
    table = dash_table.DataTable(
        data=dfm.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in dfm.columns]
    )
    return table


@app.callback(
    Output("line-chart", "figure"),
    [Input("genre2", "value")])
def update_line_chart(genre):
    df1 = genre_backup(getyear(df, [1920, 2020]), genre)
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines", x=df1["year"], y=df1["avg_vote"], name="avg_rating"))
    fig.add_trace(go.Scatter(mode="lines", x=df1["year"], y=df1["max rating"], name="max_rating"))
    fig.add_trace(go.Scatter(mode="lines", x=df1["year"], y=df1["min rating"], name="min_rating"))
    # fig.update_layout(title_text="Genre line chart ",
    #                   title_font_size=20)
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=20),
        legend=dict(orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0.005),
        xaxis=dict(
            title=dict(
                text='Year',
                font=dict(
                    color='#B90000'
                ),
            )
        ),
    )
    return fig


@app.callback(
    Output(component_id='dif-chart', component_property='figure'),
    Input(component_id='age', component_property='value'),
    Input(component_id='genre-filter', component_property='value'))
def update_age_chart(age_range, genre):
    df1 = age(getyear(df, [1920, 2020]), age_range, genre)
    fig = go.Figure()
    # fig = px.line(df1, x='year', y='male_avg_vote')
    fig.add_trace(go.Scatter(mode="lines", x=df1["year"], y=df1["male_avg_vote"], name="male_avg_rating"))
    fig.add_trace(go.Scatter(mode="lines", x=df1["year"], y=df1["female_avg_vote"], name="female_avg_rating"))
    fig.update_layout(xaxis_title='Year',
                      yaxis_title='Avg_rating')
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title=dict(
                text='Year',
                font=dict(
                    color='#B90000'
                ),
                standoff=10
            )
        ),
    )
    return fig


@app.callback(
    Output("subplot", "figure"),
    [Input("Tab_slider", "value")])
def update_subplot(Year):
    df1 = country_ranking(getyear(df, Year))
    df2 = income(getyear(df, Year))
    fig = make_subplots(rows=1, cols=2, column_width=[0.4, 0.6],
                        subplot_titles=("Top 10 Countries with the most movies", "Top 10 Box Revenue Movies"))
    fig.add_trace(
        go.Bar(
            x=df1['count'],
            y=df1['country'],
            marker=dict(
                color='rgba(50, 171, 96, 0.6)',
                line=dict(
                    color='rgba(50, 171, 96, 1.0)',
                    width=2
                ),
            ),
            orientation='h',
            # name='Top 10 Countries with the most movies',
            stream=dict(maxpoints=10)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df2['worlwide_gross_income'],
            y=df2['title'],
            marker=dict(
                color='rgb(176, 224, 230)',
                line=dict(
                    color='rgb(176, 224, 230)',
                    width=2
                ),
            ),
            orientation='h',
            # name='Top 10 rated movies '
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Film counts", title_font=dict(
        color='#B90000'), title_standoff=6, row=1, col=1)
    fig.update_xaxes(title_text="Box revenue", title_font=dict(
        color='#B90000'), title_standoff=6, row=1, col=2)
    fig.update_layout(
        height=380,
        width=1222,
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
        ),
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True,
        ),
        # legend=dict(orientation="h",
        #             yanchor="bottom",
        #             y=1.02,
        #             xanchor="left",
        #             x=0.1,
        #             itemwidth=30,
        #             font=dict(size=16)
        #             ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=35),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='black', size=17, family='Acumin,helvetica neue,sans-serif')
    return fig


@app.callback(
    Output("country_map", "figure"),
    Input("map_choose_company", "value"),
    Input("map_choose_genre", "value")
)
def update_map(company_name, genre):
    df1 = company(df)
    df2 = duration(getgenre(df, genre), company_name)
    fig = make_subplots(rows=1, cols=2, column_width=[0.45, 0.55],
                        subplot_titles=("Top 10 Company with the most movies", "Duration"))
    fig.add_trace(
        go.Bar(
            x=df1['count'],
            y=df1['company'],
            marker=dict(
                color='#FF9933',
                line=dict(
                    color='#FF9933',
                    width=2
                ),
            ),
            orientation='h',
            # name='Top 10 Countries with the most movies',
            stream=dict(maxpoints=10)
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=df2["year"], y=df2["duration"], mode='markers'), row=1, col=2)
    fig.update_layout(height=400, width=1222)
    fig.update_layout(margin={"r": 20, "t": 20, "l": 30, "b": 30}, showlegend=False)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='black', size=17, family='Acumin,helvetica neue,sans-serif')
    fig.update_xaxes(title_text="Film counts",
                     title_font=dict(
                         color='#B90000'),
                     title_standoff=6,
                     zeroline=False,
                     showline=False,
                     showticklabels=True,
                     showgrid=True,
                     row=1, col=1)
    fig.update_xaxes(title_text="Year", title_font=dict(
        color='#B90000'), title_standoff=6, row=1, col=2)
    fig.update_yaxes(
        showgrid=False,
        showline=False,
        showticklabels=True,
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text="duration",
        row=1,
        col=2,
        title_standoff=4
    )
    return fig


@app.callback(
    Output("cytoscape", "elements"),
    [Input("graph_choose_genre", "value"),
     Input("graph_choose_correlation", "value")
     ])
def update_subgraph(name, correlation):
    elements = []
    data = high_value(df)
    relationship_data = relationship(data, correlation, name)
    actors = {}
    for key, item in relationship_data.items():
        for actor in key:
            actors[actor] = actors.get(actor, 0) + 1
    for k in actors.keys():
        elements.append({'data': {'id': k, 'label': k}, 'classes': 'red'})
    for k, weight in relationship_data.items():
        elements.append({'data': {'source': k[0], 'target': k[1], 'weight': weight}, 'classes': 'red'})
    return elements


if __name__ == '__main__':
    app.run_server(debug=True)
