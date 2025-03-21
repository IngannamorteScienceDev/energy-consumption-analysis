import pandas as pd
from data_preprocessing import load_and_transform, merge_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def prepare_features(df):
    """
    Добавляет временные признаки: Month, DayOfWeek, Hour.
    """
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Hour'] = df['Datetime'].dt.hour
    return df


def main():
    # 1. Загрузка и объединение данных
    load_long, temp_long = load_and_transform()
    merged = merge_data(load_long, temp_long, zone=1, station=1)

    # 2. Подготовка признаков
    merged = prepare_features(merged)

    # 3. Удаляем строки с пропущенными значениями в признаках и целевой переменной
    features = ['temp', 'Month', 'DayOfWeek', 'Hour']
    merged = merged.dropna(subset=features + ['load'])

    # Определение X (признаки) и y (целевая переменная)
    X = merged[features]
    y = merged['load']

    # 4. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Прогнозирование и оценка модели
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    # 7. Вывод первых 10 фактических и прогнозируемых значений
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print(results.head(10))


if __name__ == '__main__':
    main()
