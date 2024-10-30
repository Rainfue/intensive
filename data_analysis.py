# %%
#!! IMPORTS !!
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# %%
# читаем .csv файл и выводим первые 10 строк
df = pd.read_csv('D:\Helper\MLBazyak\intensive\day7\main.csv')
df.head(10)
# %%
# смотрим размерность дата фрейма
df.shape
#%%
df['year_of_construction'].value_counts().index.tolist()
# %%
# изучаем типы переменных
df.info()
# %%
# составляем список переменных, 
# у которых не должно быть значение в виде float
float_list = ['floor', 'floors_count', 
              'rooms_count', 'price', 
              'heating_type', 'year_of_construction']
# %%
# собственно заменяем тип данных на int64
for var in float_list:
    df[var] = pd.to_numeric(df[var], errors='coerce').astype('Int64')
# %%
df['kitchen_meters'] = pd.to_numeric(
    df['kitchen_meters'].str.replace(',', '.').apply(lambda x: x[:-3] if pd.notna(x) else np.nan),
    errors='coerce'
).astype('float64')  # Используем Float64 для поддержки NaN

df['kitchen_meters']

# %%
df['living_meters']
# %%
df['living_meters'] = pd.to_numeric(
    df['living_meters'].str.replace(',', '.').apply(lambda x: x[:-3] if pd.notna(x) else np.nan),
    errors='coerce'
).astype('float64')  # Используем Float64 для поддержки NaN

df['living_meters']
# %%
# проверяем что все прошло успешно
df.info()
# %%
# можно заметить -1 в данных, которые обозначают их отсутствие
# поэтому заменим все -1 на Nan
df = df.replace('-1', np.nan)
df = df.replace(-1, np.nan)
df = df.replace(-1.0, np.nan)
df = df.replace('-1.0', np.nan)
df.head(10)
# %%
df['kitchen_meters'].value_counts().index.to_list()
# %%
df2 = df.copy(deep=True)
# %%
# чтобы понять, какие колонки оставлять, а какие оставлять, 
# рассмотрим процентаж пропусков в каждой из них
# поставим трешхолд 20% пропусков, и удалим каждую колонку,
# которая его не соблюдает
missing_procent = df.isna().mean()*100
missing_procent
# %%
for column, procent in missing_procent.items():
    if procent > 50:
        print(f'Колонка {column} не прошла трешхолд, удаление...')
        del df[column]
#%%
df3= df.copy(deep=True)
# колонки с ссылкой, номером дома, номером телефона, 
# автором объявления и улицей, также удаляются за ненадобностью
for column in ['url', 'author', 'street', 'house_number', 'phone']:
    del df[column]
# %%

# %%
# проверяем
df.info()
# %%
# выводим .head() и смотрим дальше
df.head()
# %%
df['total_meters'] = pd.to_numeric(df['total_meters'], errors='coerce').astype('float64')
# %%
# создаем новую переменную - цена за квадратный метр
df['price_per_meter'] = df['price']/df['total_meters']
df['price_per_meter'].head()
# %%
plt.figure(figsize=(13, 6))
df['decade'] = (df['year_of_construction'] // 10) * 10  # Преобразуем в десятилетие
df['decade'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Decade of Construction")
plt.ylabel("Number of Buildings")
plt.title("Distribution of Buildings by Decade of Construction")
plt.show()
# %%
plt.figure(figsize=(18, 6))
df.groupby('year_of_construction')['price_per_meter'].mean().sort_values(ascending=False).plot(kind='bar')
plt.xlabel("Year of Construction")
plt.ylabel("Average Price per Meter")
plt.title("Average Price per Meter by Year of Construction (Sorted by Price)")
plt.show()
# %%
df['year_of_construction'].value_counts()

# %%
df.head(10)
# %%

# %%
# %%
# выведем график пропусков
msno.bar(df)
plt.show()
# %% _________________________________________________________________
# начинаем работу с пропусками
# выводим список всех колонок, где есть пропуски
df.any()
# %%
missing_columns = df.columns[df.isnull().any()]
print("Столбцы с пропусками:", missing_columns.tolist())
# %%
# -------------------- пропуски с комнатами
df[df['rooms_count'].isnull()]
# %%
df['rooms_count'].median()
# %%
df['rooms_count'].fillna(2,inplace=True)
df[df['rooms_count'].isnull()]
# %%
# -------------------- пропуски с author_type
df[df['author_type'].isna()]
# %%
df['author_type'].mode()[0]
# %%
df['author_type'].fillna(df['author_type'].mode()[0], inplace=True)
# %%
df[df['author_type'].isna()]
# %%
# ---------------- пропуски с location
df[df['location'].isna()]
# %%
df['location'].fillna(df['location'].mode()[0], inplace=True)
# %%
df[df['location'].isna()]
# %%
# ---------------- пропуски с deal_type и accommodation_type
del df['accommodation_type']
del df['deal_type']
# %%
# -------------- пропуски с floor
df[df['floor'].isna()]
# %%
df['floor'].fillna(df['floor'].mode()[0], inplace=True)
# %%
df[df['floor'].isna()]
# %%
# --------------- пропуски с floors_count
df = df.dropna(subset=['floors_count'])
# %%
# --------------- пропуски с price
df[df['price'].isna()].shape # 78 пропуска
# %%
# создаем словари верхних и нижних границ среднего значения цены по городам
down_border = {}
up_border = {}

for city in df['location'].value_counts().index.tolist():
    # Рассчитываем статистические параметры
    q1 = df[df['location'] == city]['price'].quantile(0.15)  # Первый квартиль
    q3 = df[df['location'] == city]['price'].quantile(0.80)  # Третий квартиль

    # Записываем данные в словари
    down_border[city] = q1
    up_border[city] = q3
    
# %%
print(down_border)
print(up_border)
# %%
# Фильтрация данных по диапазону цен
def filter_prices(df, down_border, up_border):
    # Создаем маску для фильтрации
    mask = pd.Series([False] * len(df))  # Начальная маска, все значения False

    for city in down_border.keys():
        # Условия фильтрации для каждого города
        lower_bound = down_border[city]
        upper_bound = up_border[city]

        # Обновляем маску, добавляя условия для текущего города
        city_mask = (df['location'] == city) & (df['price'] >= lower_bound) & (df['price'] <= upper_bound)
        mask |= city_mask  # Объединяем с текущей маской

    # Фильтруем DataFrame, оставляя только строки с True в маске
    filtered_df = df.loc[mask | df['price'].isna()]  # Добавляем NaN в результирующий DataFrame

    return filtered_df

# %%
filtered_df = filter_prices(df, down_border, up_border)
# %%
filtered_df.head()
# %%
# заменяем NaN в price на среднее значение
for city in filtered_df['location'].value_counts().index.tolist():
    city_df = filtered_df[filtered_df['location']==city]
    missing_count = city_df['price'].isna().sum()
    
    if missing_count > 0:  # Проверяем, есть ли пропуски
        mean_price = city_df['price'].mean().astype(int)  # Вычисляем среднее значение цены для данного города
        filtered_df.loc[filtered_df['location'] == city, 'price'] = filtered_df.loc[filtered_df['location'] == city, 'price'].fillna(mean_price)  # Заменяем NaN на среднее значение
    
    print(f'{city}: {missing_count} пропусков, средняя цена: {mean_price if missing_count > 0 else "Нет пропусков"}')
# %%
# смотрим результат :)
filtered_df[filtered_df['price'].isna()].shape
# %%
filtered_df.shape
# %%
# выводим пустые столбцы уже от отфильтрованной колонки
missing_columns =filtered_df.columns[filtered_df.isnull().any()]
print("Столбцы с пропусками:", missing_columns.tolist())
# %%
# -------------------------------- пропуски в year_of_construction
filtered_df[filtered_df['year_of_construction'].isna()]
# %%
# %%
filtered_df['year_of_construction'].fillna(filtered_df['year_of_construction'].median(), inplace=True)
filtered_df[filtered_df['year_of_construction'].isna()]
# %%
# print(filtered_df['year_of_construction'].median().astype(int))

# filtered_df['year_of_construction'].fillna(2014, inplace=True)
# filtered_df[filtered_df['year_of_construction'].isna()]
# %%
filtered_df[filtered_df['total_meters'].isna()]
# %%
# -------------------------------- пропуски в price_per_meters
filtered_df['price_per_meter'] = (filtered_df['price']/filtered_df['total_meters']).round(2)
filtered_df[filtered_df['price_per_meter'].isna()]
# %%
# ---------------------------------- пропуски в living_meters
# Отбираем строки без пропусков
filtered_df['living_meters'].head(20)
# %%
df_no_na = filtered_df[filtered_df['living_meters'].notna()]
X_train = df_no_na[['total_meters', 'floor', 'floors_count', 'price', 'year_of_construction', 'price_per_meter']]
y_train = df_no_na['living_meters']

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)
# %%
# Отбор строк с пропущенными значениями в 'living_meters' для предсказания
nan_indices = filtered_df['living_meters'].isna()
X_pred = filtered_df.loc[nan_indices, ['total_meters', 'floor', 'floors_count', 'price', 'year_of_construction', 'price_per_meter']]
#%%
# Проверка, есть ли строки для предсказания
if not X_pred.empty:
    # Прогнозирование
    predicted_values = model.predict(X_pred)
    # Замена пропущенных значений на предсказанные
    filtered_df.loc[nan_indices, 'living_meters'] = predicted_values
else:
    print("Нет строк с пропущенными значениями для предсказания.")
# %%
filtered_df[filtered_df['living_meters'].isna()]
# %%
filtered_df.head(20)
# %%
# ------------------------------------ пропуски kitchen_meters
# Отбираем строки без пропусков
df_no_na = filtered_df[filtered_df['kitchen_meters'].notna()]
X_train = df_no_na[['total_meters', 'floor', 'floors_count', 'price', 'year_of_construction', 'price_per_meter', 'living_meters']]
y_train = df_no_na['kitchen_meters']

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)
# %%
# Отбор строк с пропущенными значениями в 'living_meters' для предсказания
nan_indices = filtered_df['kitchen_meters'].isna()
X_pred = filtered_df.loc[nan_indices, ['total_meters', 'floor', 'floors_count', 'price', 'year_of_construction', 'price_per_meter', 'living_meters']]
#%%
# Проверка, есть ли строки для предсказания
if not X_pred.empty:
    # Прогнозирование
    predicted_values = model.predict(X_pred)
    # Замена пропущенных значений на предсказанные
    filtered_df.loc[nan_indices, 'kitchen_meters'] = predicted_values
else:
    print("Нет строк с пропущенными значениями для предсказания.")
# %%
filtered_df[filtered_df['kitchen_meters'].isna()]
# %%
filtered_df.head(20)
# %%
# ----------------------------------------- пропуски в district и underground
# выведем колво пропусков в коэдой из колонок
print(filtered_df[filtered_df['district'].isna()].shape[0]/filtered_df.shape[0])
print(filtered_df[filtered_df['underground'].isna()].shape[0]/filtered_df.shape[0])
# %%
# Замена NaN значений в 'district' и 'underground' на значения из 'location'
filtered_df['district'].fillna(filtered_df['location'], inplace=True)
filtered_df['underground'].fillna(filtered_df['location'], inplace=True)
# %%
del filtered_df['decade']

# %%
filtered_df[filtered_df['street'].isna()]
# %%
filtered_df['street'].value_counts().index.tolist()
# %%
filtered_df = filtered_df.dropna(subset=['street'])
filtered_df[filtered_df['street'].isna()]
# %%
# выводим пустые столбцы уже от отфильтрованной колонки
missing_columns =filtered_df.columns[filtered_df.isnull().any()]
print("Столбцы с пропусками:", missing_columns.tolist())
# %%
msno.matrix(filtered_df)
plt.figure(figsize=(15,10))
plt.show()
print(filtered_df.shape)
# %%
# нам нужно перекодировать author_type и location в числовые значения, для дальнейшей работы
def number_encode_features(init_df):
    result = init_df.copy()
    encoders = {}

    for column in result.columns:
        if result.dtypes[column] == object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, column

# %%
encoded_data, encoders = number_encode_features(filtered_df)
encoded_data.head()
#%%
# скопируем наш датафрейм, и удалим все пропуски у копии

# %%
# получили относительно чистый датасет, хоть и довольно радикальным способом
encoded_data.shape
# %%
# выведем график распределения количества квартир по городам
city_counts = encoded_data['location'].value_counts()
print(city_counts)

plt.figure(figsize=(15,10))
city_counts.plot(kind='bar')
plt.title('Количество квартир в городах')
plt.xlabel('Город')
plt.ylabel('Количество квартир')
plt.xticks(rotation=80)
plt.show()
# %%
# снова посмотрим на наш датасет
encoded_data.info()
# %%

# %%
# смотрим тепловую карту наших значений
plt.figure(figsize=(20,10))
sns.heatmap(encoded_data.corr(),
            square=True,
            annot=True,
            fmt='.2g',
            cmap='Blues')
plt.savefig('heatmap.png')
# %%
encoded_data.isnull().any()
# %%
ml_df = encoded_data.copy(deep=True)
# %%
del ml_df['price']
# %%
ml_df.head(10)
ml_df.columns.tolist()
# %%
# ---------------------------------------------------------------------------------------------
X = ml_df.drop('price_per_meter', axis=1)
y = ml_df['price_per_meter']
# %%
y
# %%
X.isnull().any()
# %%
# pазделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# обучение модели
model = LinearRegression()
model.fit(X_train, y_train)
# %%
# прогнозирование
y_pred = model.predict(X_test)
# %%
# 1. MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# 2. MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 3. RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 4. R²
r_squared = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r_squared}")

# 5. MAPE (если нужно)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
# %%
errors = y_test - y_pred

# Настройка общего пространства графиков
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Ошибки против истинных значений
sns.scatterplot(x=y_test, y=errors, ax=axs[0, 0])
axs[0, 0].axhline(0, color='red', linestyle='--')
axs[0, 0].set_title('График ошибок')
axs[0, 0].set_xlabel('Истинные значения')
axs[0, 0].set_ylabel('Ошибки')

# График 2: Распределение ошибок
sns.histplot(errors, bins=30, kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Распределение ошибок')
axs[0, 1].set_xlabel('Ошибки')
axs[0, 1].set_ylabel('Частота')

# График 3: Фактические против предсказанных значений
axs[1, 0].scatter(y_test, y_pred)
axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Линия y=x
axs[1, 0].set_title('Фактические против предсказанных значений')
axs[1, 0].set_xlabel('Истинные значения')
axs[1, 0].set_ylabel('Предсказанные значения')

# График 4: Плотность ошибок по предсказанным значениям
sns.kdeplot(x=y_pred, y=errors, fill=True, cmap="viridis", ax=axs[1, 1])
axs[1, 1].axhline(0, color='red', linestyle='--')
axs[1, 1].set_title('Плотность ошибок по предсказанным значениям')
axs[1, 1].set_xlabel('Предсказанные значения')
axs[1, 1].set_ylabel('Ошибки')

# Общий вывод графиков
plt.tight_layout()
plt.show()
# %%
model = RandomForestRegressor()
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
# %%
# 1. MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# 2. MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 3. RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 4. R²
r_squared = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r_squared}")

# 5. MAPE (если нужно)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
# %%
errors = y_test - y_pred

# Настройка общего пространства графиков
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Ошибки против истинных значений
sns.scatterplot(x=y_test, y=errors, ax=axs[0, 0])
axs[0, 0].axhline(0, color='red', linestyle='--')
axs[0, 0].set_title('График ошибок')
axs[0, 0].set_xlabel('Истинные значения')
axs[0, 0].set_ylabel('Ошибки')

# График 2: Распределение ошибок
sns.histplot(errors, bins=30, kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Распределение ошибок')
axs[0, 1].set_xlabel('Ошибки')
axs[0, 1].set_ylabel('Частота')

# График 3: Фактические против предсказанных значений
axs[1, 0].scatter(y_test, y_pred)
axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Линия y=x
axs[1, 0].set_title('Фактические против предсказанных значений')
axs[1, 0].set_xlabel('Истинные значения')
axs[1, 0].set_ylabel('Предсказанные значения')

# График 4: Плотность ошибок по предсказанным значениям
sns.kdeplot(x=y_pred, y=errors, fill=True, cmap="viridis", ax=axs[1, 1])
axs[1, 1].axhline(0, color='red', linestyle='--')
axs[1, 1].set_title('Плотность ошибок по предсказанным значениям')
axs[1, 1].set_xlabel('Предсказанные значения')
axs[1, 1].set_ylabel('Ошибки')

# Общий вывод графиков
plt.tight_layout()
plt.show()
# %%
import joblib
# сохраняем модель в файл
joblib.dump(model, 'random_forest_model4.pkl')


# %%
data = {
    'model_name': ['RFR Base', 'RFR + Street + Years Deleted', 'RFR + Street + Years Filled', 'RFR + Years Deleted'],  # Названия моделей
    'mae': [34200.38, 34876.61, 34043.1, 34205.33],  # Значения MAE для каждой модели
    'r_squared': [86, 87.5, 87.3, 86.21]  # Значения R^2 для каждой модели
}

# Создание DataFrame
metrics_df = pd.DataFrame(data)
metrics_df
# %%

# Установим стиль и цветовую палитру
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))

# Создание scatter plot с Seaborn
scatter = sns.scatterplot(
    x='mae', 
    y='r_squared', 
    data=metrics_df, 
    s=150, 
    color='skyblue', 
    edgecolor='black'
)

# Добавление аннотаций
for i, txt in enumerate(metrics_df['model_name']):
    plt.annotate(
        txt, 
        (metrics_df['mae'][i], metrics_df['r_squared'][i]), 
        xytext=(5, -5), 
        textcoords='offset points', 
        fontsize=10, 
        color='black'
    )

# Оформление осей и заголовка
plt.xlabel('MAE (Mean Absolute Error)', fontsize=12)
plt.ylabel('R² (%)', fontsize=12)
plt.title('Зависимость MAE и R² для моделей', fontsize=14, weight='bold')

# Добавление сетки
plt.grid(True, linestyle='--', alpha=0.7)

# Настройка границ осей для лучшего отображения данных
plt.xlim(metrics_df['mae'].min() - 1000, metrics_df['mae'].max() + 1000)
plt.ylim(metrics_df['r_squared'].min() - 1, metrics_df['r_squared'].max() + 1)

plt.show()
# %%
test_df = pd.DataFrame()
# %%
test_df['author_type_encoded'] = encoded_data['author_type']
test_df['author_type'] = filtered_df['author_type']
test_df[['author_type_encoded', 'author_type']]
# %%
test_df['location'] = encoded_data['location']
test_df['location_encoded'] = filtered_df['location']
test_df[['location_encoded', 'location']]
# %%
filtered_df.columns.tolist()
# %%
