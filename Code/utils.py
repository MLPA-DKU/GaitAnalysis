import datetime


def dt_printer():
    return datetime.datetime.now()


def dt_printer_m(modules):
    data = list()
    data.append(dt_printer())
    for module in modules:
        data.append(module())

    return data

