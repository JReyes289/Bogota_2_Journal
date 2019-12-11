################################################################################################
#### Dash application of Team 2 - Bogota - DS4A
################################################################################################

# Load main libraries
from plotly import tools as pytools
import plotly_express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from app import app
import dash_table

# import functional libraries

from datetime import datetime as dt
import datetime 
import pandas as pd
import plotly.graph_objects as go
import json


#global df, states_grouped

####################################################################
#Import dataframes
####################################################################


########################################################################################################################################
### SQL conector
########################################################################################################################################
import psycopg2
engine = psycopg2.connect(
    database="final_db",
    user="juan",
    password="1234",
    host="nps-demo-instance.c2fezqs1nmx5.us-east-2.rds.amazonaws.com",
    port='5432'
)

token = 'pk.eyJ1IjoibmV3dXNlcmZvcmV2ZXIiLCJhIjoiY2o2M3d1dTZiMGZobzMzbnp2Z2NiN3lmdyJ9.cQFKe3F3ovbfxTsM9E0ZSQ'

with open('colombia.geo.json',encoding="latin-1") as f:
    
    geojson = json.loads(f.read())

sql3 = """
SELECT *
FROM color_cuantia;
"""
df = pd.read_sql(sql3, con=engine)

############################################################# init layout ################################################################

# Layout of Dash App
layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.H2("ANTICORRUPTION CONTRACT APP"),


##############################################  start selectors ############################################



                        html.Div([
                            html.H4("Date range"),
                            #className="div-for-dropdown",
                            #children=[
                                dcc.DatePickerRange(
                                    id='date-range',
                                    min_date_allowed = dt(2013,1,1),
                                    max_date_allowed = dt(2019,12,31),
                                    start_date =df['fecha_ini_ejec_contrato'].min(),
                                    display_format= 'DD MMM YYYY', 
                                    end_date=df['fecha_ini_ejec_contrato'].max(),
                                    #calendar_orientation='vertical',
                                )],
                            #],
                        ),
                        # Change to side-by-side for mobile layout
                        html.Div([
                            #className="row",
                            #children=[
                                ############## selector draw ############

                                dcc.RadioItems(
                                    id="classifier_select",
                                    options=[
                                        {"label": "Quantity", "value": "Q"},
                                        {"label": "Number", "value": "N"},
                                    ],
                                    value='Q',
                                    labelStyle={'display': 'inline-block', 'margin': '12px'}
                                ),
                            ]),





##############################################  end selectors ##############################################


                                #html.H4(id="total-quantity"),
                                #html.H4(id="total-contracts"),

                                #html.Br(),                                
                                
                                ##################### tabla de datos ###########################

                                html.Div(
                                    #className="six columns card",
                                    #children=[
                                        dash_table.DataTable(
                                            id='table_departments',
                                            columns=[
                                                {'name': 'Department', 'id': 'departamento_entidad'},
                                                {'name': 'Number', 'id': 'num_contract'},
                                                {'name': 'Amount', 'id': 'cuantia_contrato'},
                                            ],
                                            style_cell_conditional=[
                                                {
                                                    'if': {'column_id': 'departamento_entidad'},
                                                    'textAlign': 'left'
                                                }
                                            ],
                                            style_cell={'width': '50px', 'border': '1px solid black' },
                                            style_table={
                                                'maxHeight': '475px',
                                                'overflowY': 'scroll'
                                            },
                                            style_header={
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'fontWeight': 'bold',
                                                'textAlign': 'center'
                                            }                                            
                                        )
                                    #]
                                ),

                                html.Br(),

                                html.Div([
                                    html.Div([
                                        html.H5(id="total-quantity"),
                                        ],
                                        className="one-third column",
                                    ),
                                    html.Div([
                                        html.H5(id="total-contracts"),
                                        ],
                                        className="one-half column",
                                    ),
                                ],
                                    id="header_selectors",
                                    className="row flex-display",
                                    #style={"margin-bottom": "25px"},
                                ),


                                html.Br(),

                                html.Div([
                                    dcc.Markdown(
                                        children=["Source: [datos SECOP](https://www.contratos.gov.co/consultas/resultadoListadoProcesos.jsp#)"]),
                                    ],                        
                                    className="one-third column",
                                ),   
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(
                            id='map-plot',
                            figure={ 
                                #'data': [go.Scattermapbox()],
                                'data': [go.Choroplethmapbox()],

                                'layout': go.Layout(
                                        width=1250,
                                        height=800, 
                                        mapbox_style="streets",
                                        mapbox_accesstoken=token,
                                        mapbox_zoom=5,
                                        margin={'t': 0, 'l': 0, 'r': 35, 'b': 0},
                                        mapbox_center={"lat": 4.6097102, "lon": -74.081749}
                                    )
                            }
                        ),
                          ####################################################################  
                    ],
                ),
            ],
        )
    ]
)



########### end layout ##################################


########################################################################
# Callback to update the map's colours
########################################################################

@app.callback(
    [
        dash.dependencies.Output('map-plot', 'figure'),
        dash.dependencies.Output("total-quantity", "children"),
        dash.dependencies.Output("total-contracts", "children"),
        dash.dependencies.Output('table_departments', 'data')
    ], # component with id map-plot will be changed, the 'figure' argument is updated
    [
        dash.dependencies.Input('date-range', 'start_date'), # input with id date-picker-range and the start_date parameter
        dash.dependencies.Input('date-range', 'end_date'),
        dash.dependencies.Input('classifier_select', 'value'),
    ]
)
def update_colours_map(start_date, end_date, value):

    s_date = pd.to_datetime(start_date).date()
    e_date = pd.to_datetime(end_date).date()
    tmp_colour = df[(df['fecha_ini_ejec_contrato'] >= s_date) & (df['fecha_ini_ejec_contrato'] <= e_date)]
    cleaned_df = tmp_colour.drop(['longitud', 'latitud'], axis=1)
    total_quantity = 'Total amount (Bs.): COP $ {:.4f}'.format(tmp_colour['cuantia_contrato'].sum()/1000000000000)
    total_contracts = 'Total number: {:d}'.format(int(tmp_colour['num_contract'].sum()))
    table_grouped = cleaned_df.groupby(['departamento_entidad'], as_index=False).sum()
    states_grouped = cleaned_df.groupby(['state_abbr'], as_index=False).sum()
    states_grouped['cuantia_contrato'] = states_grouped['cuantia_contrato']/1000000000
    states_grouped['latitud'] = tmp_colour['latitud'].unique()
    states_grouped['longitud'] = tmp_colour['longitud'].unique()
    states_grouped['Coordinate'] = states_grouped[['latitud', 'longitud']].apply(lambda x: 'Coordinate lat {:.4f}, lon {:.4f}'.format(x[0], x[1]), axis=1)

    if value == 'Q':
        tmp_z=states_grouped['cuantia_contrato']
        tmp_colorbar_title="KM $COP"
    else:
        tmp_z=states_grouped['num_contract']
        tmp_colorbar_title="Quantity"
    

    return { 
            'data' : [go.Choroplethmapbox(
                z = tmp_z,
                colorbar_title = tmp_colorbar_title,                        
                geojson=geojson,
                locations=states_grouped['state_abbr'],
                #colorscale='Viridis',
                text=states_grouped['Coordinate'],



                colorscale=[[0, "rgb( 247, 220, 111 )"],
                            #[1./10000, "rgb( 241, 196, 15 )"],
                            [1./100, "rgb( 52, 152, 219 )"],
                            #[1./100, "rgb(  41, 128, 185 )"],
                            [1./10, "rgb(30, 132, 73 )"],
                            [1, "rgb( 20, 90, 50 )"]],

        
        )],
            'layout': go.Layout(
                #width=1250,
                #height=750, 
                mapbox_style="streets",
                mapbox_accesstoken=token,
                mapbox_zoom=5,
                margin={'t': 0, 'l': 0, 'r': 35, 'b': 0},
                mapbox_center={"lat": 4.6097102, "lon": -74.081749}
            )
        }, total_quantity, total_contracts, table_grouped.sort_values('cuantia_contrato',ascending=False).to_dict('records')

