from plotly import tools as pytools
import plotly_express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import plotly.graph_objects as go

from app import app

####################################################################
#Import dataframes
####################################################################
list_departments = ['Amazonas', 'Antioquia', 'Arauca', 'Atlántico', 'Bogotá D.C.', 'Bolívar', 'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 
                    'Cauca', 'Cesar', 'Chocó', 'Cundinamarca', 'Córdoba', 'Guainía', 'Guaviare', 'Huila', 'La Guajira', 'Magdalena', 'Meta', 
                    'Nariño', 'Norte De Santander', 'Putumayo', 'Quindío', 'Risaralda', 'San Andrés, Providencia y Santa Catalina', 'Santander', 
                    'Sucre', 'Tolima', 'Valle del Cauca', 'Vaupés', 'Vichada']

list_nivel = ['TERRITORIAL', 'NACIONAL', 'No Definida']
nivel_entidad = pd.read_csv("nivel_entidad.csv")
dict_nivel_entidad = {'NACIONAL': 0, 'No Definida': 1, 'TERRITORIAL': 2}

orden_entidad = pd.read_csv("orden_entidad.csv")
list_orden_entidad = [x for x in orden_entidad.orden_entidad_original]
dict_orden_entidad = {'DISTRITO CAPITAL': 0,  'NACIONAL CENTRALIZADO': 1,
  'NACIONAL DESCENTRALIZADO': 2,  'No Definido': 3,  'TERRITORIAL DEPARTAMENTAL CENTRALIZADO': 4,  'TERRITORIAL DEPARTAMENTAL DESCENTRALIZADO': 5,
  'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 1': 6,  'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 2': 7,'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 3': 8,  'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 4': 9,'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 5': 10,  'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 6': 11}

municipio_entrega = pd.read_csv("municipio_entrega.csv",encoding="latin-1",sep=";")
list_municipio_entrega= [x for x in municipio_entrega.municipio_entrega_original]
dict_municipio_entrega = municipio_entrega.set_index("municipio_entrega_original").to_dict()['municipio_entrega']

municipio_obtencion = pd.read_csv("municipio_obtencion.csv",encoding="latin-1",sep=";")
list_municipio_obtencion = [x for x in municipio_obtencion.municipio_obtencion_original]
dict_municipio_obtencion = municipio_obtencion.set_index("municipio_obtencion_original").to_dict()['municipio_obtencion']

municipios_ejecucion = pd.read_csv("municipios_ejecucion.csv",encoding="latin-1",sep=";")
list_municipios_ejecucion = [x for x in municipios_ejecucion.municipios_ejecucion_original]
dict_municipio_ejecucion = municipios_ejecucion.set_index("municipios_ejecucion_original").to_dict()['municipios_ejecucion']

latitud=pd.read_csv("data_latitud.csv")
dict_latitud = latitud.set_index("nombre").to_dict()["latitud"]
dict_longitud = latitud.set_index("nombre").to_dict()["longitud"]

filename = 'Fmodel_1M_Smote_OK.sav'
loaded_model = pickle.load(open(filename, 'rb'))


layout = html.Div(children=[
    html.Div(
        className="row app-body",
        children=[            
            # User Controls
            html.Div(
                className="twelve columns",
                children=[

                    html.Div(
                        className="padding-top-bot row",
                        children=[
                            html.Div(
                                className="two columns",
                                children=[
                                    html.H6("Select Department",),
                                    dcc.Dropdown(
                                        id="department_dropdown-score",
                                        #multi=True,
                                        options = [{'label': i, 'value': i}
                                        for i in list_departments],                                            
                                        value=list_departments[0],
                                    ),
                                ]),

                                html.Div(
                                    className='three columns',
                                    children=[
                                        html.H6("Select a Date Range for start and end of contract"),
                                        dcc.DatePickerRange(
                                            id="date-range-score",
                                            min_date_allowed = dt(2013,1,1),
                                            max_date_allowed = dt(2019,12,31),
                                            display_format='DD MMM YYYY',
                                            start_date=dt(2019,1,1),
                                            end_date=dt(2019,12,31)
                                        ),
                                ]),
                                
                                html.Div(
                                    id="Predicted-Score-div",
                                    className="two columns indicator pretty_container",
                                    children=[
                                        html.P(id="Predicted_Score", className="indicator_value"),
                                        html.P('Predicted Score', className="twelve columns indicator_text"),
                                ]),
                                
                            ],
                        )]
                    ),

########################### Additional dropdowns ################################################################


            html.Div(
                className="padding-top-bot row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("Select Entity Level"),
                            dcc.Dropdown(
                                id="level_dropdown-score",
                                options = [{'label': i, 'value': i}
                                for i in list_nivel ],                                            
                                #multi=True,
                                #placeholder="Select Entity level",
                                value=list_nivel[0]
                            ),
                        ]
                    ),
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("Select Entity territorial"),
                            dcc.Dropdown(
                                id="level_dropdown1-score",
                                options = [{'label': i, 'value': i}
                                for i in list_orden_entidad ],                                            
                                #multi=True,
                                #placeholder="Select Entity level",
                                value=list_orden_entidad[0]
                            ),
                        ]
                    ),
                ]
            ),


            html.Div(
                className="padding-top-bot row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("contract obtaining municipality"),
                            dcc.Dropdown(
                                id="level_dropdown2-score",
                                options = [{'label': i, 'value': i}
                                for i in list_municipio_entrega ],                                            
                                #multi=True,
                                #placeholder="Select contract obtaining municipality",
                                value=list_municipio_obtencion[0]
                            ),
                        ]
                    ),
                ]
            ),

            html.Div(
                className="padding-top-bot row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("contract delivery municipality"),
                            dcc.Dropdown(
                                id="level_dropdown3-score",
                                options = [{'label': i, 'value': i}
                                for i in list_municipio_entrega ],                                            
                                #multi=True,
                                #placeholder="contract delivery municipality",
                                value=list_municipio_entrega[0]
                            ),
                        ]
                    ),
                ]
            ),

            html.Div(
                className="padding-top-bot row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("contract execution municipality"),
                            dcc.Dropdown(
                                id="level_dropdown4-score",
                                options = [{'label': i, 'value': i}
                                for i in list_municipios_ejecucion ],                                            
                                #multi=True,
                                #placeholder="contract execution municipality",
                                value=list_municipios_ejecucion[0]
                            ),
                        ]
                    ),
                ]
            ),

            html.Div(
                className="padding-top-bot row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("Input contract value"),
                            dcc.Input(
                            id="input_range-score", type="number", value=100000,
                            min=0, 
                            ),
                        ]
                    ),
                ]
            ),
            
            html.Div(
                className="padding-top-bot row",
                children=[
                    html.Div(
                        className="two columns",
                        children=[
                            html.H6("Input value of contract additions"),
                            dcc.Input(
                            id="input_range-add-score", type="number", value=200,
                            min=0, 
                            ),
                        ]
                    ),
                ]
            ),

            html.Div(
                className="twelve columns",
                children=[
                    html.Div(
                        className="padding-top-bot row",
                        children=[
                            html.Div(
                                className="six columns",
                                style={'margin-right': '15px', 'margin-left': '0px'},
                                children=[
                                    html.Div(children='''Total amount.'''),
                                    dcc.Graph(
                                        id='amount_graph-score',
                                        figure={
                                            'data': [],
                                            'layout': {
                                                #'title': 'Dash Data Visualization'
                                            }
                                        }

                                    ),
                                ]
                            ),
                        
                        ],
                    )]
                ),


########################### END parche histogramas ################################################################3

        ]
        )
    ])

@app.callback(                              ###HISTOGRAMAS ENTIDAD

    
        dash.dependencies.Output("amount_graph-score", "figure"),
        #dash.dependencies.Output('Predicted_Score', 'children'),
    
    (    
        dash.dependencies.Input('date-range-score', 'start_date'),
        dash.dependencies.Input('date-range-score', 'end_date'),
        dash.dependencies.Input('department_dropdown-score', 'value'),
        dash.dependencies.Input('level_dropdown-score', 'value'),
        dash.dependencies.Input('level_dropdown1-score', 'value'),
        dash.dependencies.Input('level_dropdown2-score', 'value'),
        dash.dependencies.Input('level_dropdown3-score', 'value'),
        dash.dependencies.Input('level_dropdown4-score', 'value'),
        dash.dependencies.Input('input_range-score', 'value'),
        dash.dependencies.Input('input_range-add-score', 'value'),
        
            ),
)
def update_score(start_date, end_date, value_department, Entity_level,Entity_territorial_level,
                municipality_obtaintion,municipality_delivery,municipality_execution,contract_value,
                contractAddition_value):
    
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    #print(contract_value)
    Log_cuantia_contrato_prediccion=np.log(float(contract_value))
    Log_cuantiaaddition_contrato_prediccion=np.log(float(contractAddition_value))
    duracion_prediccion= (end_date-start_date)/ np.timedelta64(1, 'D')
    Numeroanno_firma_del_contrato_prediccion=start_date.year
    mes_fin_ejec_contrato_prediccion=end_date.month
    quarter_fin_ejec_contrato_prediccion=end_date.quarter
    Anno_fin_ejec_contrato_prediccion=end_date.year
    latitud_prediccion=dict_latitud[value_department]
    longitud_prediccion=dict_longitud[value_department]
    nivel_entidad_prediccion=dict_nivel_entidad[Entity_level]
    orden_entidad_prediccion=dict_orden_entidad[Entity_territorial_level]
    municipio_obtencion_prediccion=dict_municipio_obtencion[municipality_obtaintion]
    municipio_entrega_prediccion=dict_municipio_entrega[municipality_delivery]
    quarter_fin_contrato_departamento_ejecucion_prediccion=str(value_department)+"-"+str(quarter_fin_ejec_contrato_prediccion)
    municipios_ejecucion_prediccio=dict_municipio_ejecucion[municipality_execution]
    
    
    df_prediccion = pd.DataFrame([[Log_cuantia_contrato_prediccion,Log_cuantiaaddition_contrato_prediccion,duracion_prediccion,
    Numeroanno_firma_del_contrato_prediccion,mes_fin_ejec_contrato_prediccion,quarter_fin_ejec_contrato_prediccion,
    Anno_fin_ejec_contrato_prediccion,latitud_prediccion,longitud_prediccion,nivel_entidad_prediccion,
    orden_entidad_prediccion,municipio_obtencion_prediccion,municipio_entrega_prediccion,5,municipios_ejecucion_prediccio]], 
    columns=['Log_cuantia_contrato', 'Log_valor_total_de_adiciones', 'duracion',
       'Numeroanno_firma_del_contrato', 'mes_fin_ejec_contrato',
       'quarter_fin_ejec_contrato', 'Anno_fin_ejec_contrato', 'latitud',
       'longitud', 'nivel_entidad', 'orden_entidad', 'municipio_obtencion',
       'municipio_entrega', 'quarter_fin_contrato_departamento_ejecucion',
       'municipios_ejecucion'])

    animals=['Sancionado', 'No sancionado']

    probabilidad_sancion= loaded_model.predict_proba(df_prediccion)[0][1]
    probabilidad_no_sancion= loaded_model.predict_proba(df_prediccion)[0][0]

    #print(probabilidad_sancion)
    fig = go.Figure([go.Bar(x=animals, y=[probabilidad_sancion, probabilidad_no_sancion])])
    fig.update_layout(
    title='sanctioned contract',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Probability',
        titlefont_size=16,
        tickfont_size=14,
    ))
    return fig