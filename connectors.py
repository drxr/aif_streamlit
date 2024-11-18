import pandas as pd
import psycopg2
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
        self.connection = psycopg2.connect(f"dbname={self.name} user={self.user} host={self.host} port={self.port} password={self.password}")
    
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
