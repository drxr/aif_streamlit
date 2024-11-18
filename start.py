# загружаем библиотеки
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from connectors import SQLConnector
from analysis import PaymentsFileAnalysis


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
