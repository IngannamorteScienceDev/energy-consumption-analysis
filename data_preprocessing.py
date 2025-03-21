import pandas as pd


def load_and_transform():
    """
    Импортирует данные из CSV-файлов, преобразует их из широкого формата в длинный,
    создает столбец Datetime и сортирует данные по времени.

    Возвращает:
      load_long  - преобразованный датафрейм с данными нагрузки,
      temp_long  - преобразованный датафрейм с температурными данными.
    """
    # Чтение исходных файлов
    load_data = pd.read_csv('data/Load_history.csv')
    temp_data = pd.read_csv('data/temperature_history.csv')

    # Преобразование данных нагрузки: wide → long
    load_long = pd.melt(load_data,
                        id_vars=['zone_id', 'year', 'month', 'day'],
                        value_vars=[f'h{i}' for i in range(1, 25)],
                        var_name='hour',
                        value_name='load')
    # Убираем запятые в числовых значениях и преобразуем load к float
    load_long['load'] = load_long['load'].astype(str).str.replace(',', '').astype(float)
    # Удаляем символ "h" и приводим столбец hour к целому числу
    load_long['hour'] = load_long['hour'].str.replace('h', '').astype(int)

    # Преобразование данных температуры: wide → long
    temp_long = pd.melt(temp_data,
                        id_vars=['station_id', 'year', 'month', 'day'],
                        value_vars=[f'h{i}' for i in range(1, 25)],
                        var_name='hour',
                        value_name='temp')
    # Удаляем символ "h" и приводим столбец hour к целому числу
    temp_long['hour'] = temp_long['hour'].str.replace('h', '').astype(int)

    # Создание базового столбца Datetime из year, month и day
    load_long['Datetime'] = pd.to_datetime(load_long[['year', 'month', 'day']])
    temp_long['Datetime'] = pd.to_datetime(temp_long[['year', 'month', 'day']])

    # Прибавление часов к базовой дате
    load_long['Datetime'] = load_long['Datetime'] + pd.to_timedelta(load_long['hour'], unit='h')
    temp_long['Datetime'] = temp_long['Datetime'] + pd.to_timedelta(temp_long['hour'], unit='h')

    # Сортировка данных по Datetime
    load_long = load_long.sort_values(by='Datetime')
    temp_long = temp_long.sort_values(by='Datetime')

    return load_long, temp_long


def merge_data(load_long, temp_long, zone=1, station=1):
    """
    Фильтрует данные по указанной зоне (load) и станции (temperature),
    а затем объединяет их по столбцу Datetime.

    Параметры:
      zone    - номер зоны для данных нагрузки (по умолчанию 1)
      station - номер станции для температурных данных (по умолчанию 1)

    Возвращает:
      merged  - объединенный датафрейм
    """
    load_zone = load_long[load_long['zone_id'] == zone]
    temp_station = temp_long[temp_long['station_id'] == station]
    merged = pd.merge(load_zone, temp_station, on='Datetime', how='inner')
    return merged


if __name__ == '__main__':
    # Тестовый запуск для проверки работы модуля
    load_long, temp_long = load_and_transform()
    merged = merge_data(load_long, temp_long)
    print("Объединенные данные:")
    print(merged.head())
