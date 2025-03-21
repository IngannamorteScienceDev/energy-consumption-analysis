import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1. Импорт данных
    load_data = pd.read_csv('data/Load_history.csv')
    temp_data = pd.read_csv('data/temperature_history.csv')

    # Вывод первых строк для проверки
    print("=== Первые строки данных нагрузки ===")
    print(load_data.head(), "\n")

    print("=== Первые строки данных температуры ===")
    print(temp_data.head(), "\n")

    # 2. Информация о датафреймах и проверка пропущенных значений
    print("=== Информация о данных нагрузки ===")
    print(load_data.info(), "\n")

    print("=== Информация о данных температуры ===")
    print(temp_data.info(), "\n")

    print("=== Пропущенные значения в данных нагрузки ===")
    print(load_data.isna().sum(), "\n")

    print("=== Пропущенные значения в данных температуры ===")
    print(temp_data.isna().sum(), "\n")

    # 3. Преобразование данных из широкого в длинный формат (melt)
    # Для нагрузки: идентификаторы – 'zone_id', 'year', 'month', 'day'
    load_long = pd.melt(load_data,
                        id_vars=['zone_id', 'year', 'month', 'day'],
                        value_vars=[f'h{i}' for i in range(1, 25)],
                        var_name='hour',
                        value_name='load')

    # Убираем запятые и преобразуем load к float
    load_long['load'] = load_long['load'].astype(str).str.replace(',', '').astype(float)
    # Удаляем символ "h" и приводим столбец hour к целому числу
    load_long['hour'] = load_long['hour'].str.replace('h', '').astype(int)

    # Для температуры: идентификаторы – 'station_id', 'year', 'month', 'day'
    temp_long = pd.melt(temp_data,
                        id_vars=['station_id', 'year', 'month', 'day'],
                        value_vars=[f'h{i}' for i in range(1, 25)],
                        var_name='hour',
                        value_name='temp')

    # Удаляем символ "h" и приводим столбец hour к целому числу
    temp_long['hour'] = temp_long['hour'].str.replace('h', '').astype(int)

    # 4. Создание столбца Datetime (векторизированно)
    # Создаем базовую дату из year, month и day
    load_long['Datetime'] = pd.to_datetime(load_long[['year', 'month', 'day']])
    temp_long['Datetime'] = pd.to_datetime(temp_long[['year', 'month', 'day']])

    # Прибавляем часы. Если hour равен 24, прибавление 24 часов корректно переводит дату на следующий день.
    load_long['Datetime'] = load_long['Datetime'] + pd.to_timedelta(load_long['hour'], unit='h')
    temp_long['Datetime'] = temp_long['Datetime'] + pd.to_timedelta(temp_long['hour'], unit='h')

    # Сортировка по Datetime для корректного построения графиков
    load_long = load_long.sort_values(by='Datetime')
    temp_long = temp_long.sort_values(by='Datetime')

    # 5. Визуализация распределений и поиск выбросов
    # Гистограмма нагрузки
    plt.figure(figsize=(10, 5))
    plt.hist(load_long['load'], bins=50, color='blue', edgecolor='black')
    plt.title('Распределение нагрузки')
    plt.xlabel('Load')
    plt.ylabel('Частота')
    plt.show()

    # Boxplot нагрузки
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=load_long['load'])
    plt.title('Boxplot нагрузки')
    plt.xlabel('Load')
    plt.show()

    # Гистограмма температуры
    plt.figure(figsize=(10, 5))
    plt.hist(temp_long['temp'], bins=50, color='orange', edgecolor='black')
    plt.title('Распределение температуры')
    plt.xlabel('Temperature')
    plt.ylabel('Частота')
    plt.show()

    # Boxplot температуры
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=temp_long['temp'])
    plt.title('Boxplot температуры')
    plt.xlabel('Temperature')
    plt.show()

    # 6. Построение временных рядов
    # Временной ряд нагрузки
    plt.figure(figsize=(12, 5))
    plt.plot(load_long['Datetime'], load_long['load'], label='Load', color='blue')
    plt.title('Временной ряд нагрузки')
    plt.xlabel('Datetime')
    plt.ylabel('Load')
    plt.legend()
    plt.show()

    # Временной ряд температуры
    plt.figure(figsize=(12, 5))
    plt.plot(temp_long['Datetime'], temp_long['temp'], label='Temperature', color='orange')
    plt.title('Временной ряд температуры')
    plt.xlabel('Datetime')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
