import os
from Code.dynamic_library import column_info
import pandas as pd
import csv
from datetime import datetime


directory = '../Result'
# directory = '/home/blackcow/mlpa/workspace/gait-rework/gait-rework/Result'


class DataStore:
    def __init__(self, info, savedir):
        self.column = info
        # self.df = pd.DataFrame(columns=info)
        self.store_dict = dict()
        self.savedir = savedir
        self.store_init()
        self.file_name = None

    def store_init(self):
        self.store_dict['datetime'] = list()
        self.store_dict['method'] = list()
        self.store_dict['model'] = list()
        self.store_dict['samples'] = list()
        self.store_dict['repeat'] = list()
        self.store_dict['accuracy'] = list()
        self.store_dict['loss'] = list()

    def stock_result(self, df_param):
        if len(df_param) is 4:
            self.store_dict['datetime'].append(df_param[0])
            self.store_dict['method'].append('')
            self.store_dict['model'].append('')
            self.store_dict['samples'].append('')
            self.store_dict['repeat'].append(df_param[1])
            self.store_dict['accuracy'].append(df_param[3])
            self.store_dict['loss'].append(df_param[2])
        else:
            self.store_dict['datetime'].append(df_param[0])
            self.store_dict['method'].append(df_param[1])
            self.store_dict['model'].append(df_param[2])
            self.store_dict['samples'].append(df_param[3])
            self.store_dict['repeat'].append(df_param[4])
            self.store_dict['accuracy'].append(df_param[6])
            self.store_dict['loss'].append(df_param[5])

    def find_result(self, idx):
        return self.df[idx + 1]

    def get_column(self):
        return self.column

    def save_result_obo(self, param, df_param):

        def set_result(input_df):
            storage = dict()
            if len(input_df) is 4:
                storage['datetime'] = [str(input_df[0])]
                storage['method'] = ['']
                storage['model'] = ['']
                storage['samples'] = ['']
                storage['repeat'] = [str(input_df[1])]
                storage['accuracy'] = [str(input_df[3])]
                storage['loss'] = [str(input_df[2])]
            else:
                storage['datetime'] = [str(input_df[0])]
                storage['method'] = [str(input_df[1])]
                storage['model'] = [str(input_df[2])]
                storage['samples'] = [str(input_df[3])]
                storage['repeat'] = [str(input_df[4])]
                storage['accuracy'] = [str(input_df[6])]
                storage['loss'] = [str(input_df[5])]
            return storage
        result_info = set_result(df_param)

        file_name = f"{param.model_name}_{param.method}_{param.folder}_" \
                    f"{datetime.today().strftime('%y%m%d')}_blackbox.csv"
        if os.path.exists(self.savedir) is not True:
            os.mkdir(self.savedir)

        fit_path = self.savedir
        for add_path in ['experiment', f'{param.model_name}', f'{param.method}']:
            save_path = os.path.join(fit_path, add_path)
            if os.path.exists(save_path) is not True:
                os.mkdir(save_path)
            fit_path = save_path
        save_path = os.path.join(fit_path, file_name)

        if os.path.isfile(save_path) is False:
            df = pd.DataFrame.from_dict(self.store_dict)
            df.to_csv(save_path, sep=',', na_rep='NaN', index=False)
        else:
            already = pd.read_csv(save_path)
            df = pd.DataFrame.from_dict(result_info)
            frame = pd.concat([already, df], ignore_index=False, sort=False)
            frame.to_csv(save_path, sep=',', na_rep='NaN', index=False)

    def save_result(self, param):
        file_name = f"{param.model_name}_{param.method}_{param.folder}" \
                    f"_{datetime.today().strftime('%y%m%d')}.csv"
        if os.path.exists(self.savedir) is not True:
            os.mkdir(self.savedir)

        fit_path = self.savedir
        for add_path in ['experiment', f'{param.model_name}', f'{param.method}']:
            save_path = os.path.join(fit_path, add_path)
            if os.path.exists(save_path) is not True:
                os.mkdir(save_path)
            fit_path = save_path
        save_path = os.path.join(fit_path, file_name)
        df = pd.DataFrame.from_dict(self.store_dict)
        df.to_csv(save_path, sep=',', na_rep='NaN')

        # if os.path.isfile(save_path) is False:
        #     df = pd.DataFrame.from_dict(self.store_dict)
        #     df.to_csv(save_path, sep=',', na_rep='NaN')
        # else:
        #     already = pd.read_csv(save_path)
        #     df = pd.DataFrame.from_dict(self.store_dict)
        #     frame = pd.concat([already, df], ignore_index=False, sort=False)
        #     frame.to_csv(save_path, sep=',', na_rep='NaN')

