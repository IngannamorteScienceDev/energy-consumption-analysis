import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_and_transform, merge_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def prepare_time_features(df):
    """
    Добавляет временные признаки: месяц, день недели, час.
    """
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Hour'] = df['Datetime'].dt.hour
    return df


def create_lag_features(df, lag=24):
    """
    Добавляет лаговый признак 'load_lag_<lag>' – значение нагрузки за <lag> часов до текущего момента.
    Предполагается, что df отсортирован по Datetime.
    """
    df = df.sort_values('Datetime').copy()
    df[f'load_lag_{lag}'] = df['load'].shift(lag)

    # Удаляем строки, в которых после сдвига получились NaN
    df.dropna(subset=[f'load_lag_{lag}'], inplace=True)
    return df


def time_based_split(df, date_split='2007-01-01'):
    """
    Разделение набора данных на train и test по дате.
    Все данные до date_split – train, после (и включая) date_split – test.
    """
    train = df[df['Datetime'] < date_split].copy()
    test = df[df['Datetime'] >= date_split].copy()
    return train, test


def main():
    # 1. Загрузка и объединение данных
    load_long, temp_long = load_and_transform()
    merged = merge_data(load_long, temp_long, zone=1, station=1)

    # 2. Добавляем временные признаки
    merged = prepare_time_features(merged)

    # Удаляем строки с пропущенными значениями в исходных признаках (temp, load)
    merged.dropna(subset=['temp', 'load'], inplace=True)

    # 3. Добавляем лаговый признак (нагрузка за 24 часа до текущего момента)
    merged = create_lag_features(merged, lag=24)

    # 4. Повторно удаляем возможные NaN (на случай, если появились ещё)
    merged.dropna(subset=['temp', 'load', 'load_lag_24'], inplace=True)

    # 5. Делим данные на train и test по дате
    # Предположим, что train – данные до 2007-01-01, test – с 2007-01-01 и позже
    train, test = time_based_split(merged, '2007-01-01')

    # 6. Определяем список признаков и целевую переменную
    features = ['temp', 'Month', 'DayOfWeek', 'Hour', 'load_lag_24']
    X_train, y_train = train[features], train['load']
    X_test, y_test = test[features], test['load']

    # 7. Обучение моделей
    # 7.1 Random Forest
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"RandomForestRegressor - MSE: {rf_mse:.2f}, R2: {rf_r2:.2f}")

    # Важность признаков
    importances = rf.feature_importances_
    print("Feature importances (Random Forest):")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.3f}")

    # 7.2 Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42, n_estimators=100)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)

    gb_mse = mean_squared_error(y_test, gb_preds)
    gb_r2 = r2_score(y_test, gb_preds)
    print(f"GradientBoostingRegressor - MSE: {gb_mse:.2f}, R2: {gb_r2:.2f}")

    # 8. Сравнение фактических и предсказанных значений во времени
    # Формируем общий датафрейм результатов
    results = test[['Datetime', 'load']].copy()
    results.rename(columns={'load': 'Actual'}, inplace=True)
    results['RF_Pred'] = rf_preds
    results['GB_Pred'] = gb_preds

    # Сортируем по времени
    results.sort_values(by='Datetime', inplace=True)

    # 9. Построим график только для первых 500 точек тестовой выборки
    subset = results.head(500)

    plt.figure(figsize=(12, 5))
    plt.plot(subset['Datetime'], subset['Actual'], label='Actual', marker='o', linestyle='--')
    plt.plot(subset['Datetime'], subset['RF_Pred'], label='RF_Pred', marker='x')
    plt.plot(subset['Datetime'], subset['GB_Pred'], label='GB_Pred', marker='d')
    plt.xlabel('Datetime')
    plt.ylabel('Load')
    plt.title('Сравнение фактического и предсказанного (первые 500 точек теста)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
