import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st
from datetime import datetime
import io


pio.templates.default = "plotly_white"


class PaymentsFileAnalysis:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
    
    def make_segments(self) -> None:

        '''
        Метод для построения RFM анализа
        '''

        pays = self.data[self.data.status == 'Paid']

        # определяем период жизни абонентов (разница между максимальной и минимальной датой)
        pays['period'] = (
            pays.groupby('user_id')['order_date']
            .transform(lambda cell: int((cell.max() - cell.min()) / pd.Timedelta('1d')) + 1)
            )

        # определяем дату анализа (максимальная дата + 1 день) и выводим на экран
        now = pays.order_date.max() + pd.Timedelta('1d')
        st.write(f'Дата отсчета для RFM анализа: {now}')
            
        # определяем РФМ значения
        pays['R_value'] = pays.order_date.apply(lambda cell: int(((now - cell) / pd.Timedelta('1d'))))
        pays['M_value'] = pays.groupby('user_id')['price'].transform('sum')
        pays['F_value'] = pays.groupby('user_id')['user_id'].transform('count') / pays.period

        # бьем R на ранги
        r_bins = [0, 90, 365, pays.R_value.max()]
        r_labels = [3, 2, 1]
        pays['R'] = pd.cut(pays.R_value, bins=r_bins, labels=r_labels)

        # бьем F на ранги
        f_bins = [0, .027, .11, pays.F_value.max()]
        f_labels = [1, 2, 3]
        pays['F'] = pd.cut(pays.F_value, bins=f_bins, labels=f_labels)

        # бьем М на ранги
        m_bins = [-0.1, 600, 2800, pays.M_value.max()]
        m_labels = [1, 2, 3]
        pays['M'] = pd.cut(pays.M_value, bins=m_bins, labels=m_labels)

        # Сцепляем ранги в общий ранг RFM
        pays['RFM'] = pays.R.astype('str') + pays.F.astype('str') + pays.M.astype('str')

        # st.markdown позволяет выводить текст с разметкой как в юпитере
        st.markdown('**Описание сегментов**')
        st.markdown('''
        Итого получаем таблицу с идентификаторами пользователей и сегментами RFM (просьба заказчика). Метрики расчитывались следующим образом: 1 - плохо, 2 - терпимо, 3 - хорошо
                    ''')
        st.markdown('''
        **R:** **1** - был давно (больше 365 дней), **2** - не заходил от 90 до 365 дней, **3** - был недавно (не более 90 дней с момента визита)  
        **F:** **1** - заходит не часто (не более 1 раза за 38 дней), **2** - средняя активность(1 заход в течение 9 - 38 дней),  **3** - высокая активность (чаще 1 раза в 9 дней)   
        **M:** **1** - жадные (до 600 рублей за все время), **2** - средняя сумма (600-2800 рублей), **3** - щедрые (более 2800 рублей)
                    ''')
        st.markdown('''
        |R - recency|F - frequency|M - monetary|
        |:--|:--|:--|
        |Время отсутствия|Частота заходов|Количество денег|
        |**1** - был более 365 дней назад|**1** - заходит не чаще 0.027 раз в день|**1** - заплатил менее 600 рублей за все время|
        |**2** - был от 365 до 90 дней назад|**2** - заходит от 0.027 до 0.11 раз в день|**2** - заплатил от 600 до 2800 рублей за все время|
        |**3** - был менее 90 дней назад|**3** - заходит чаще 0.11 раз в день|**3** - заплатил более 2800 рублей за все время|
                    ''')
        
        st.write('---')
        
        st.markdown('**Сводная таблица по результатам RFM-анализа**')
        # готовим таблицу для RFM сегментов    
        rfm_stats = pays.groupby('RFM').agg({'user_id':'nunique', 'price': ['count', 'mean', 'sum']}).reset_index()
            
        rfm_stats.columns = ['RFM сегмент', 'Человек в сегменте, чел.',
                            'Количество пожертвований, ед.', 'Среднее пожертвование, руб.',
                            'Сумма пожертвований, руб.']
        rfm_stats['test'] = rfm_stats['Сумма пожертвований, руб.']

        # готовим данные для сводной итоговой таблицы    
        test_df = pays[['order_date', 'price', 'user_id', 'RFM']]
        test_df['period'] = pd.to_datetime(test_df.order_date).dt.to_period('M')
        test_df = test_df.groupby(['RFM', 'period'])['price'].sum().reset_index()
        test_df = test_df.sort_values(by='period')
            
        rfm_stats = rfm_stats.merge(test_df.groupby('RFM')['price'].apply(list).reset_index(), left_on='RFM сегмент', right_on='RFM')
        rfm_stats = rfm_stats.drop(columns=['RFM'])

        # выводим на экран итоговую таблицу
        rfm_stats['Человек в сегменте, чел.'] =  rfm_stats['Человек в сегменте, чел.'].astype('float')
        rfm_stats['Количество пожертвований, ед.'] = rfm_stats['Количество пожертвований, ед.'].astype('float')
        rfm_stats['Сумма пожертвований, руб.'] = rfm_stats['Сумма пожертвований, руб.'].astype('float')
        rfm_stats['test'] = rfm_stats['test'].astype('float')

        st.dataframe(rfm_stats.set_index('RFM сегмент'),
                column_config={
                "test": st.column_config.ProgressColumn(  # этот код делает колонку с прогресс баром
                    "Сумма донаций",
                    help="Общая сумма донаций",
                    format="%f руб.",
                    min_value=0,
                    max_value=rfm_stats.test.max(),
                ),
                "price": st.column_config.BarChartColumn(  # этот код добавляет бар чарт в ячейки колонки
                    "Платежи по месяцам",
                    help="The sales volume in the last 6 months",
                    y_min=0,
                    y_max=500000,
                ),
            },
            height=980)

        # делаем датасет с идентифкатором пользователя и сегментом RFM
        final_rfm = pays[['user_id', 'RFM']]

        st.write('---')
        st.markdown('**Скачать файл с сегментами пользователей**')
        # добавляем кнопку для загрузки csv файла с пользователями по РФМ сегментам
        st.download_button(
            label="Скачать доноров с RFM сегментом",
            data=final_rfm.to_csv(sep=';').encode('utf-8'),
            file_name="rfm_users.csv",
            mime="text/csv",
            )

    def make_cohorts(self) -> None:

        '''
        Метод для построения когортного анализа: отток, средний чек и LTV
        '''

        # для анализа берем только успешные платежи
        df_file = self.data[self.data.status == 'Paid']
        
        # добавляем столбец с месяцем платежа
        df_file['invoice_month'] = df_file['order_date'].apply(lambda cell: datetime(cell.year, cell.month, 1))
        # добавляем месяц когорты (первая покупка пользователя)
        df_file['cohort_month'] = df_file.groupby('user_id')['invoice_month'].transform('min')
        # добавляем периода жизни (разница в месяцах между месяцем платежа и месяцем когорты)
        df_file['period'] = ((df_file['invoice_month'] - df_file['cohort_month']) / pd.Timedelta('30d')).astype('int')
        # добавляем столбец с суммой месячных трат пользователя
        df_file['month_total'] = df_file.groupby(['user_id', 'period'])['price'].transform('sum')

        #Считаем активных пользователей по когортам (берем март 2023 года, студенты могут сделать элемент выбора даты начала анализа)
        retention = df_file[df_file.cohort_month >= '2023-03-01'].groupby(['cohort_month', 'period'])['user_id'].apply(pd.Series.nunique)
        
        # Делаем сводную таблицу оттока по когортам (уникальные пользователи)
        retention = retention.reset_index()
        retention.cohort_month = retention.cohort_month.dt.date
        retention_pivot = retention.pivot(index='cohort_month',columns='period',values='user_id')
        # добавляем размер когорт в сводную таблицу
        cohort_size = retention_pivot.iloc[:,0]
        retention_final = retention_pivot.divide(cohort_size,axis=0)

        # отрисовываем теплокарту с оттоком пользователей в когортах
        retention_figure, ax = plt.subplots(figsize=(15, 8))
        plt.title('Когортный анализ: retention rate', fontdict=dict(weight='bold', size=14))
        sns.heatmap(data=retention_final, annot=True, fmt='.1%', vmin=0, vmax=.3, linewidth=.3, cmap="Blues", cbar=False, ax=ax)
        plt.grid(False)
        st.pyplot(retention_figure)

        # делаем сводную таблицу среднего чека по когортам
        mean_pay_data = df_file[df_file.cohort_month >= '2023-03-01'].groupby(['cohort_month', 'period'])['month_total'].mean().reset_index()
        mean_pay_data.cohort_month = mean_pay_data.cohort_month.dt.date
        average_payments = mean_pay_data.pivot(index='cohort_month', columns='period', values='month_total')

        # отрисовываем теплокарту среднего чека по когортам
        check_figure, ax = plt.subplots(figsize=(15, 8))
        plt.title('Когортный анализ: средний чек', fontdict=dict(weight='bold', size=14))
        sns.heatmap(data=average_payments, annot = True, vmin = 0, vmax=3000, linewidth=.3, cmap="Blues", cbar=False, fmt='.2f', ax=ax)
        plt.grid(False)
        st.pyplot(check_figure)

        # отрисовываем LTV по когортам (кумулятивная сумма среднего чека по периодам жизни когорты)
        ltv_figure, ax = plt.subplots(figsize=(15, 8))
        plt.title('Когортный анализ: LTV', fontdict=dict(weight='bold', size=14))
        sns.heatmap(data=average_payments.cumsum(axis=1), annot = True, vmin = 0, vmax=23000, linewidth=.3, cmap="Blues", cbar=False, fmt='.0f', ax=ax)
        plt.grid(False)
        st.pyplot(ltv_figure)

    def make_common_info(self) -> None:

        '''
        Метод для вывода общей информации о деятельности фонда:
        - общее количество платежей
        - динамика платежей
        - количество рекурентов
        - количество отказов
        - ошибки платежей
        - прочая информация (делают студенты)
        '''

        # разделяем платежи на успех, отказ и ошибка
        pays = self.data[self.data.status == 'Paid']
        unpays = self.data[self.data.status == 'notpaid']
        fails = self.data[self.data.status == 'fail']

        # выводим подзаголовок
        st.subheader('Общая информация')

        # выводим на экран количество счетов
        st.write(f'Всего оплаченных счетов: **{pays.shape[0]}** счетов')
        st.write(f'Сумма оплаченных счетов: **{pays.price.sum():,}** рублей')
        mean_pay = pays.groupby('order_date')['price'].mean().reset_index()
        st.write(f'Средний чек: **{pays.price.sum()/ pays.shape[0]:.2f}** рублей')
        
        st.write('---')

        # выводим подзаголовок
        st.subheader('Динамика платежей')

        # рисуем график со средними платежами за день
        fig_mean = px.line(mean_pay, x="order_date", y="price", title='Динамика среднего дневного пожертвования, руб.')
        st.plotly_chart(fig_mean)

        # группируем платежи по дням
        pays_line = pays.groupby('order_date')['price'].sum().reset_index()

        # рисуем график с суммой платежей по дням
        fig_dinamics = go.Figure()
        fig_dinamics.add_traces(go.Scatter(x=pays_line.order_date, 
                                        y=pays_line.price, 
                                        line=dict(color="lightgrey"),
                                        mode='lines', name = 'Платежи'))
        fig_dinamics.add_traces(go.Scatter(x=pays_line.order_date, 
                                        y=pays_line.price.rolling(15).mean(), 
                                        line=dict(color="crimson", width=2),
                                        mode='lines', name = 'Платежи скользящее среднее'))
        fig_dinamics.update_layout(title_text="Динамика всех пожертвований по дням, руб.",
            legend=dict(yanchor="top",
                                    y=0.99,
                                    xanchor="left", x=0.01,
                                    orientation='h'))
        st.plotly_chart(fig_dinamics)

        # рисуем график с суммой платежей по месяцам
        pays['month'] = pd.to_datetime(pays.order_date).dt.to_period('M')
        month_pays = pays.groupby('month')['price'].sum().reset_index()
        month_pays['month'] = month_pays.month.astype('str')
        fig_months = px.bar(month_pays, y='price', x='month', text_auto='.2s',
                title="Помесячная сумма пожертвований, руб.")
        fig_months.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_months)

        st.write('---')

        # Неоплаченные счета и сбои платежей
        st.subheader('Неоплаченные пожертвования и ошибки в платежах')

        st.write(f'Всего неоплаченных счетов: **{unpays.shape[0]}**')
        st.write(f'Сумма неоплаченных счетов: **{unpays.price.sum():,}** рублей')
        st.write('---')
        st.write(f'Всего оплат с ошибкой: **{fails.shape[0]}**')
        st.write(f'Сумма оплат с ошибкой: **{fails.price.sum():,}** рублей')
        # min_date, max_date = pays.order_date.min(), pays.order_date.max()

        # График с ошибками в платежах
        fig_mistakes = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig_mistakes.add_trace(go.Pie(labels=['Оплачено', 'Не оплачено', 'Ошибка'], 
                                    values=[pays.price.sum(), unpays.price.sum(), fails.price.sum()], 
                                    name="В рублях"), 1, 1)
        fig_mistakes.add_trace(go.Pie(labels=['Оплачено', 'Не оплачено', 'Ошибка'], 
                                    values=[pays.price.count(), unpays.price.count(), fails.price.count()], 
                                    name="Количество"), 1, 2)

        fig_mistakes.update_traces(hole=.6, hoverinfo="label+percent+name")

        fig_mistakes.update_layout(
            title_text="Доля неоплаченных пожертвований и ошибок платежей",
            annotations=[dict(text='В рублях', x=0.16, y=0.5, font_size=20, showarrow=False),
                    dict(text='Количество', x=0.87, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig_mistakes)

        st.write('---')
        
        # добавляем подзаголовок
        st.subheader('Информация о рекурентах')

        # выделяем рекурентов и не рекурентов
        recs = pays[pays.recurrent == True]
        unrecs = pays[pays.recurrent == False]

        # информация по платежам и количеству рекурентов
        st.write(f'Количество рекурентов за период: **{recs.user_id.nunique()}** человек')
        st.write(f'Сумма пожертвования рекурентов за период: **{recs.price.sum():,}** рублей')
        st.write(f'Среднее пожертвование рекурентов за период: **{recs.price.mean():,.2f}** рублей')
        
        # строим график по рекурентам
        fig_rec = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig_rec.add_trace(go.Pie(labels=['Рекурент', 'Не рекурент'], 
                                    values=[recs.price.sum(), unrecs.price.sum()], 
                                    name="В рублях"), 1, 1)
        fig_rec.add_trace(go.Pie(labels=['Рекурент', 'Не рекурент'], 
                                    values=[recs.price.count(), unrecs.price.count()], 
                                    name="Количество"), 1, 2)

        fig_rec.update_traces(hole=.6, hoverinfo="label+percent+name")
        fig_rec.update_layout(
            title_text="Доля рекурентов по количеству и платежам в общем объеме пожертвований",
            annotations=[dict(text='В рублях', x=0.15, y=0.5, font_size=20, showarrow=False),
                    dict(text='Количество', x=0.82, y=0.5, font_size=20, showarrow=False)])
        
        st.plotly_chart(fig_rec)

        # buffer_1 = io.BytesIO()
        # fig_rec.write_image(file=buffer_1, format="pdf")

        # st.download_button(
        #     label="Скачать график рекуренты в формате PDF",
        #     data=buffer_1,
        #     file_name="figure.pdf",
        #     mime="application/pdf",
        # )

        st.write('---')