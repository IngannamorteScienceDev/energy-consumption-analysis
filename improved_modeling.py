import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_and_transform, merge_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def prepare_features(df):
    """
    Добавляет временные признаки: месяц, день недели, час.
    """
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Hour'] = df['Datetime'].dt.hour
    return df


def main():
    # 1. Загрузка и объединение данных
    load_long, temp_long = load_and_transform()
    # По умолчанию выбираем зону 1 для нагрузки и станцию 1 для температуры
    merged = merge_data(load_long, temp_long, zone=1, station=1)

    # 2. Подготовка признаков
    merged = prepare_features(merged)

    # Удаляем строки с пропущенными значениями в признаках и целевой переменной
    features = ['temp', 'Month', 'DayOfWeek', 'Hour']
    merged = merged.dropna(subset=features + ['load'])

    # Определяем X (признаки) и y (целевая переменная)
    X = merged[features]
    y = merged['load']

    # 3. Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Обучение модели Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    print("RandomForestRegressor - MSE: {:.2f}, R2: {:.2f}".format(rf_mse, rf_r2))

    # 5. Обучение модели Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_preds)
    gb_r2 = r2_score(y_test, gb_preds)
    print("GradientBoostingRegressor - MSE: {:.2f}, R2: {:.2f}".format(gb_mse, gb_r2))

    # 6. Визуальное сравнение предсказаний
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label="Фактические", marker='o', linestyle='--')
    plt.plot(rf_preds, label="RF Прогноз", marker='x')
    plt.plot(gb_preds, label="GB Прогноз", marker='d')
    plt.legend()
    plt.title("Фактическое vs Прогнозируемое")
    plt.xlabel("Набор данных (индексы)")
    plt.ylabel("Load")
    plt.show()


if __name__ == '__main__':
    main()
