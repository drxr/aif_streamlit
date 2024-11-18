# загружаем библиотеки
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from analysis import PaymentsFileAnalysis
from sqlalchemy import create_engine
import streamlit as st


class SQLConnector:

    '''
    Класс SQL коннектор создает подключение к базе данных и методы,
    которые позволяют работать с базой данных: запись данных в базу,
    чтение базы, внесение изменений

    Аттрибуты:
    - db_name: str - название базы данных
    - db_user: str - пользователь базы данных
    - db_host: str - хост / адрес базы данных
    - db_port: str - порт подключения базы данных
    - db_password: str - пароль для доступа к базе данных

    Методы: 
    - read_sql_query(query, engine): получение информации из базы данных через запросы
    - upload_data(data, table, engine): выгрузка датасетов в базу данных
    '''

    def __init__(self, db_name: str, db_user: str, db_host: str, db_port: str, db_password: str):
        self.name = db_name
        self.user = db_user
        self.host = db_host
        self.port = db_port
        self.password = db_password
        self.connection = create_engine(f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}')
    
    @st.cache_data()
    def read_sql_query(_self, query: str) -> pd.DataFrame:

        '''
        Метод для чтения данных из базы данных путем SQL запросов,
        выгружается информации в виде датафрейма Pandas

        Аргументы:
        - query: str - запрос к базе данных
        - engine: psycopg2.connect - коннектор PostgreSQL к базе данных
        '''
        
        return pd.read_sql(query, con=_self.connection)


    def upload_data(self, data: pd.DataFrame, table: str, engine: psycopg2.connect) -> None:

        '''
        Метод для выгрузки полученных датафреймов в базу данных SQL

        Аргументы: 
        - table: str - название таблицы в базе данных куда будут занесены данные
        - engine: psycopg2.connect - коннектор к базе данных PostgreSQL
        '''

        return data.to_sql(table, engine=self.connection, if_exists='replace')


# настраиваем опции
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
st.set_page_config(layout="wide")  # отображение элементов streamlit на всю ширину страницы

# загружаем файлы окружения
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

if os.path.exists(dotenv_path):
    load_dotenv()

# данные для подключения к базе данных
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_password = os.getenv('DB_PASSWORD')


# заголовок приложения
st.header('Аналитическая панель АиФ Доброе сердце')

# боковая панель (название)
st.sidebar.subheader("Выберите опцию:")

db_con = SQLConnector(db_name, db_user, db_host, db_port, db_password)

actions = db_con.read_sql_query('SELECT * FROM users')
action_columns = ['action_system_template_name', 'action_datetime', 'channel_name', 
                  'channel_campaign', 'channel_utm_source', 'utm_medium', 'utm_content', 
                  'utm_term', 'user_id', 'mailing_action', 'mailing_name']
actions_clean = actions[action_columns]

payments = db_con.read_sql_query('SELECT * FROM payments')
payments.columns = ['mindbox_id', 'first_action_id', 'order_date', 'channel_id',
                    'channel_external_id', 'channel_name', 'delivery_cost', 
                    'price', 'website_id', 'recurrent', 'repayment',
                    'product_id', 'product_name', 'quantity', 'price_per_item',
                    'price_of_line', 'status', 'line_number', 'line_line_number', 
                    'user_id', 'date_update']

if payments is not None:
    # вводная часть
    st.markdown('**Общая информация**')
    st.markdown(f'Загружен файл с платежами в количестве **{payments.shape[0]}** единиц с **{payments.shape[1]}** признаками за период **{payments.order_date.min()}** по **{payments.order_date.max()}**')
    st.markdown(f'Анализируемый период по платежам: **{(payments.order_date.max() - payments.order_date.min()).days}** дней')
    #конец вводной части

    rfm_cohorts = PaymentsFileAnalysis(payments)

    rfm_button = st.sidebar.button('RFM анализ')

    if rfm_button:
        rfm_cohorts.make_segments()

    cohort_button = st.sidebar.button('Когортный анализ')

    if cohort_button:
        rfm_cohorts.make_cohorts()

if actions_clean is not None:
    st.markdown(f'Загружен файл с действиями в количестве **{actions_clean.shape[0]}** единиц с **{actions_clean.shape[1]}** признаками за период **{actions_clean.action_datetime.min()}** по **{actions_clean.action_datetime.max()}**')
    st.markdown(f'Анализируемый период по действиям: **{(actions_clean.action_datetime.max() - actions_clean.action_datetime.min()).days}** дней')
    marketing_button = st.sidebar.button('Маркетинговый анализ')

if payments is not None and actions_clean is not None:

    common_button = st.sidebar.button('Сводная информация')
    if common_button:
        rfm_cohorts.make_common_info()
