import pandas as pd
from io import BytesIO
import itertools
import numpy as np
import torch
import pandahouse as ph
from app.helpers.db.clickhouse_connector import ClickhouseConnector
import asyncio
from app.config.settings import get_settings


settings = get_settings()


class OvenTimeOptimizationModel:

    def __init__(self):
        self.code_dict = {'preparing': 0, 'nagrev': 1, 'otzhig': 2, 'kovka': 3, 'prokat': 4, 'wait_time_to_change': 5, 'downtime': 6}
        self.non_specific_operations = ['nagrev', 'otzhig']
        self.time_to_chane_place = 15
        self.time_to_heat = 120

    def get_time_requared(self, task):
        total_time = 0
        for action in task['operations']:
            total_time += action['timing']
        return total_time + self.time_to_chane_place + self.time_to_heat * (len(task['operations']) - 2)

    @staticmethod
    def get_task_operation(task):

        operations = set([oper['name'] for oper in task['operations']])
        if 'kovka' in operations:
            return 'kovka'
        elif 'prokat' in operations:
            return 'prokat'
        elif 'otzhig' in operations:
            return 'otzhig'
        else:
            return None

    @staticmethod
    def get_index_of_zero_subarray(row, time_requared, availability_matrix):

        row_list = availability_matrix[row]
        intervals = [(x[0], len(list(x[1]))) for x in itertools.groupby(row_list)]

        starting_minute = 0
        for idx, interval in enumerate(intervals):
            if interval[0] == 0 and interval[1] >= time_requared:
                if idx == 0:
                    return starting_minute
                else:
                    for i in range(idx):
                        starting_minute += intervals[i][1]
                return starting_minute
        return -1

    def get_operation_code(self, operation):
        return self.code_dict[operation['name']]

    def update_result_matrix(self, result_matrix, task, task_idx, starting_idx, oven_idx, requared_heating=False):

        if requared_heating:
            result_matrix[oven_idx][starting_idx: starting_idx + self.time_to_heat] = [(task_idx, 0)] * self.time_to_heat
            starting_idx += self.time_to_heat

        for i, operation in enumerate(task['operations']):

            # обрабатываем операцию
            op_code = self.get_operation_code(operation)
            op_time = operation['timing']
            result_matrix[oven_idx][starting_idx: starting_idx + op_time] = [(task_idx, op_code)] * op_time
            starting_idx += op_time

            # добавляем промежуточные манипуляции
            if i < len(task['operations']) - 1:
                if i == 0:  # перемещение детали после нагрева
                    result_matrix[oven_idx][starting_idx: starting_idx + self.time_to_chane_place] = [(task_idx,
                                                                                                   5)] * self.time_to_chane_place
                    starting_idx += self.time_to_chane_place

                else:  # повторный нагрев детали между ковками или прокатами
                    result_matrix[oven_idx][starting_idx: starting_idx + self.time_to_heat] = [(task_idx, 1)] * self.time_to_heat
                    starting_idx += self.time_to_heat

        return result_matrix

    @staticmethod
    def get_metric(available_ovens, result_matrix):
        total = 0
        for i in range(len(available_ovens)):
            df = pd.DataFrame(list(result_matrix[i]), columns=[0, 1])
            df_1 = df[(df[0].isna()) | (df[1] == 0)]
            total += (1 - df_1.shape[0] / df.shape[0])
        return total / len(available_ovens)

    @staticmethod
    def get_wrong_temp_ovens(data):
        work_temp = []
        for i in data['series']:
            work_temp.append(i['temperature'])
        max_temp = max(work_temp)

        original_ovens = data['ovens']
        available_ovens = {i: oven for i, oven in enumerate(original_ovens)}

        collect_id = []
        for i in range(len(available_ovens)):
            if available_ovens[i]['start_temp'] > max_temp:
                collect_id.append(i)

        not_available_ovens = {}
        for j in range(len(collect_id)):
            number = collect_id[j]
            data_number = available_ovens[number]
            not_available_ovens[number] = data_number
        return not_available_ovens

    @staticmethod
    def get_siamese_oven(oven_1, task_operation, available_ovens):
        if task_operation in ['nagrev', 'otzhig']:
            return None
        for oven in available_ovens:
            if 'prokat' in available_ovens[oven]['operations'] or 'kovka' in available_ovens[oven]['operations'] and oven_1 != oven:
                return oven

    def update_availability_matrix(self, availability_matrix, task, starting_idx, oven_idx, requared_heating=False):
        if requared_heating:
            starting_idx += self.time_to_heat
        for i, operation in enumerate(task['operations']):
            # обрабатываем операцию
            op_time = operation['timing']
            if operation['name'] in ['prokat', 'kovka']:
                availability_matrix[oven_idx][starting_idx: starting_idx + op_time] = 1
            starting_idx += op_time

            # добавляем промежуточные манипуляции
            if i < len(task['operations']) - 1:
                if i == 0:  # перемещение детали после нагрева
                    starting_idx += self.time_to_chane_place
                else:  # повторный нагрев детали между ковками или прокатами
                    starting_idx += self.time_to_heat

    def epsilon_greedy_algorithm_one_day_v2(self, data, epsilon=0.0, one_brigade=True):

        wrong_temp_ovens = self.get_wrong_temp_ovens(data)

        # определяем вспомогательные объекты
        original_ovens = data['ovens']
        available_ovens = {i: oven for i, oven in enumerate(original_ovens)}
        series = data['series']

        # availability_matrix = np.zeros((len(available_ovens), 1440)).astype('int')

        shape = torch.empty(len(available_ovens), 1440)
        availability_matrix = torch.zeros_like(shape)

        result_matrix = np.empty((), dtype=object)
        result_matrix[()] = (None, None)
        result_matrix = np.full((len(available_ovens), 1440), result_matrix, dtype=object)
        availability_matrix = availability_matrix.numpy().astype('int')

        # проходим по всем сериям
        for i, task in enumerate(series):

            time_requared = self.get_time_requared(task)
            temp_requared = task['temperature']
            task_operation = self.get_task_operation(task)

            flag = True
            rnd = np.random.random()

            if flag:
                # проверяем разогретые до нужной температуры печи
                for oven in available_ovens:

                    oven_temp = available_ovens[oven]['start_temp']
                    oven_operations = available_ovens[oven]['operations']

                    if oven_temp == temp_requared and (task_operation in oven_operations or not task_operation):
                        idx = self.get_index_of_zero_subarray(oven, time_requared, availability_matrix)

                        if idx >= 0:
                            if one_brigade:
                                siamese_oven = self.get_siamese_oven(oven, task_operation, available_ovens)
                                if siamese_oven:
                                    self.update_availability_matrix(availability_matrix, task, idx, siamese_oven)
                            availability_matrix[oven][idx: idx + time_requared] = 1
                            result_matrix = self.update_result_matrix(result_matrix, task, i, idx, oven)
                            flag = False
                            break

            if rnd < epsilon and flag:
                # проверяем остальные печи
                time_requared += self.time_to_heat
                for oven in available_ovens:
                    oven_temps = available_ovens[oven]['working_temps']
                    oven_operations = available_ovens[oven]['operations']

                    if temp_requared in oven_temps and (task_operation in oven_operations or not task_operation):

                        idx = self.get_index_of_zero_subarray(oven, time_requared, availability_matrix)
                        if idx >= 0:
                            if one_brigade:
                                siamese_oven = self.get_siamese_oven(oven, task_operation, available_ovens)
                                if siamese_oven:
                                    self.update_availability_matrix(availability_matrix, task, idx, siamese_oven,
                                                               requared_heating=True)
                            availability_matrix[oven][idx: idx + time_requared] = 1
                            result_matrix = self.update_result_matrix(result_matrix, task, i, idx, oven,
                                                                 requared_heating=True)
                            available_ovens[oven]['start_temp'] = temp_requared
                            flag = False
                            break

            if rnd >= epsilon and flag:

                # проверяем печи с невостребованными температурами
                time_requared += self.time_to_heat
                for oven in wrong_temp_ovens:
                    oven_temps = available_ovens[oven]['working_temps']
                    oven_operations = available_ovens[oven]['operations']

                    if temp_requared in oven_temps and (task_operation in oven_operations or not task_operation):

                        idx = self.get_index_of_zero_subarray(oven, time_requared, availability_matrix)
                        if idx >= 0:
                            if one_brigade:
                                siamese_oven = self.get_siamese_oven(oven, task_operation, available_ovens)
                                if siamese_oven:
                                    self.update_availability_matrix(availability_matrix, task, idx, siamese_oven,
                                                               requared_heating=True)
                            availability_matrix[oven][idx: idx + time_requared] = 1
                            result_matrix = self.update_result_matrix(result_matrix, task, i, idx, oven,
                                                                 requared_heating=True)
                            available_ovens[oven]['start_temp'] = temp_requared
                            wrong_temp_ovens.pop(oven)
                            flag = False
                            break

                # проверяем остальные печи
                if flag:
                    for oven in available_ovens:

                        oven_temps = available_ovens[oven]['working_temps']
                        oven_operations = available_ovens[oven]['operations']

                        if temp_requared in oven_temps and (task_operation in oven_operations or not task_operation):

                            idx = self.get_index_of_zero_subarray(oven, time_requared, availability_matrix)
                            if idx >= 0:
                                if one_brigade:
                                    siamese_oven = self.get_siamese_oven(oven, task_operation, available_ovens)
                                    if siamese_oven:
                                        self.update_availability_matrix(availability_matrix, task, idx, siamese_oven,
                                                                   requared_heating=True)
                                availability_matrix[oven][idx: idx + time_requared] = 1
                                result_matrix = self.update_result_matrix(result_matrix, task, i, idx, oven,
                                                                     requared_heating=True)
                                available_ovens[oven]['start_temp'] = temp_requared
                                flag = False
                                break

        metric = self.get_metric(available_ovens, result_matrix)

        return result_matrix, metric

    @staticmethod
    async def make_report(date: str) -> BytesIO:
        async with ClickhouseConnector() as client:
            result = await client.fetch(
                f"""
                    select * from RL_MODELS.series_upload_ovens
                    where toDate(dt) = '{date}'
                """
            )
            if len(result) > 0:
                cols = list(result[0].keys())
                data = pd.DataFrame([dict(zip(cols, el.values())) for el in result])
                df = data[data['id_serie'] != -1]
                df_result = df.sort_values(['id_oven', 'id_serie', 'dt'], ascending=False).groupby(
                    ['id_oven', 'id_serie', 'operation_serie']) \
                    .agg({'dt': ['min', 'max']}) \
                    .rename(columns={'min': 'start', 'max': 'end'}) \
                    .reset_index()

                df_result = df_result.sort_values(['id_oven', 'id_serie', ('dt', 'start')], ascending=True)
                df_result.columns = ['_'.join(col) for col in df_result.columns]
                excel_file = BytesIO()
                df_result.to_excel(excel_file, index=False, sheet_name="Sheet1")
                excel_file.seek(0)
                return excel_file

    def save_result(self, data, result):
        series = data["data"]["series"]
        temp_series = {i: s["temperature"] for i, s in enumerate(series)}
        result_dfs = []
        for i in range(len(result)):
            result_work_alg = list(result[i])
            df = pd.DataFrame(result_work_alg, columns=["id_serie", "operation_code"])
            df["id_serie"] = df.id_serie.fillna(-1).astype(int)
            df["operation_code"] = df.operation_code.fillna(6).astype(int)
            df["operation_serie"] = df.operation_code.map(
                {v: k for k, v in self.code_dict.items()}
            )
            df["id_oven"] = i
            df["dt"] = pd.date_range(data["date"], periods=1440, freq="Min")
            df['date'] = df['dt'].dt.date
            df["temp_serie"] = df.id_serie.map(temp_series).fillna(0).astype(int)
            result_dfs.append(df)
        result_dfs = pd.concat(result_dfs)
        asyncio.run(
            self.__save_data_to_clickhouse(
                result_dfs,
                settings.db_oven_time_optimization,
                settings.table_series_upload_ovens
            )
        )
        return result_dfs

    def start_calc(self, data):
        result, metric = self.epsilon_greedy_algorithm_one_day_v2(data["data"], one_brigade=data["one_brigad"])
        self.save_result(data, result)
        output = {"result": "Данные успешно сохранены", "metric": metric}
        return output

    @staticmethod
    async def __save_data_to_clickhouse(data, db_name, table_name):
        """
        Save data to db
        """
        connection = await ClickhouseConnector(db=db_name).connect()
        result = ph.to_clickhouse(
            data, table_name, index=False, chunksize=100000, connection=connection
        )
        return result
