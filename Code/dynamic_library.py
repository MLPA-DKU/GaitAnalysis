# csv column info
csv_columns_names = {
    "datasets": ["dates",
                 "p1(L)", "p2(L)", "p3(L)", "p4(L)", "p5(L)", "p6(L)", "p7(L)", "p8(L)",
                 "ACC_X(L)", "ACC_Y(L)", "ACC_Z(L)", "GYZO_X(L)", "GYZO_Y(L)", "GYZO_Z(L)",
                 "p1(R)", "p2(R)", "p3(R)", "p4(R)", "p5(R)", "p6(R)", "p7(R)", "p8(R)",
                 "ACC_X(R)", "ACC_Y(R)", "ACC_Z(R)", "GYZO_X(R)", "GYZO_Y(R)", "GYZO_Z(R)"],
    "pressure": ["p1(L)", "p2(L)", "p3(L)", "p4(L)", "p5(L)", "p6(L)", "p7(L)", "p8(L)",
                 "p1(R)", "p2(R)", "p3(R)", "p4(R)", "p5(R)", "p6(R)", "p7(R)", "p8(R)"],
    "acc": ["ACC_X(L)", "ACC_Y(L)", "ACC_Z(L)", "ACC_X(R)", "ACC_Y(R)", "ACC_Z(R)"],
    "gyro": ["GYZO_X(L)", "GYZO_Y(L)", "GYZO_Z(L)", "GYZO_X(R)", "GYZO_Y(R)", "GYZO_Z(R)"],
    "left_pressure": ["p1(L)", "p2(L)", "p3(L)", "p4(L)", "p5(L)", "p6(L)", "p7(L)", "p8(L)"],
    "right_pressure": ["p1(R)", "p2(R)", "p3(R)", "p4(R)", "p5(R)", "p6(R)", "p7(R)", "p8(R)"]
}

# construct lib
method_select = {
    "repeat": ['smdpi', 'mdpi', 'half', 'dhalf', 'MCCV', 'base', 'sleaveone'],
    "people": 'LeaveOne',
    "CrossValidation": ['7CV', 'SCV'],
    "specific": ['sleaveone']
}

# result_collector
column_info = ['datetime', 'method', 'model', 'samples', 'repeat', 'accuracy', 'loss']

# method collector
method_info = {
    'specific': ['cropping', 'convert'],
    '5columns': ['div_base'],
    '4columns': ['BasicNet', 'ResNet', 'VGG'],
    '3columns': ['base', 'lstm', 'bi-lstm', 'lstm_attention', 'cnn_lstm', 'similarity', 'base_v2', 'ensemble'],
    '2columns': ['lgbm'],
    'vector': ['div_vec']
}

# model compactor
model_info = {
    'dl': ['BasicNet', 'ResNet', 'VGG', 'pVGG', 'base', 'lstm', 'bi-lstm', 'cnn_lstm', 'base_v2', 'div_base'],
    'c_dl': ['similarity', 'lstm_attention'],
    'v_dl': ['div_vec'],
    'ml': ['lgbm'],
    'ensemble': ['ensemble']
}