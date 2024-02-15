#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform
from time import sleep

import numpy as np
import optuna
import pandas as pd
import pyodbc
import requests
import xgboost as xgb
import yaml
from optuna.samplers import TPESampler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

start_time = datetime.datetime.now()
warnings.filterwarnings("ignore")

print("# skm_ecmwf Start! #", start_time)

# Общий раздел

# Коэффициент завышения прогноза:
OVERVALUE_COEFF = 1
# Задаем переменные (даты для прогноза и список погодных параметров)
DATE_BEG = (datetime.datetime.today() + datetime.timedelta(days=1)).strftime(
    "%d.%m.%Y"
)
DATE_END = (datetime.datetime.today() + datetime.timedelta(days=3)).strftime(
    "%d.%m.%Y"
)
DATE_BEG_PREDICT = (
    datetime.datetime.today() + datetime.timedelta(days=1)
).strftime("%Y-%m-%d")
DATE_END_PREDICT = (
    datetime.datetime.today() + datetime.timedelta(days=2)
).strftime("%Y-%m-%d")

COL_PARAMETERS = [
    "HRES_10U_FRCST_GEOOBJ",
    "HRES_10V_FRCST_GEOOBJ",
    "HRES_DEWP_FRCST_GEOOBJ",
    "HRES_TEMP_FRCST_GEOOBJ",
    "HRES_CAPE_FRCST_GEOOBJ",
    "HRES_CDIR_FRCST_GEOOBJ",
    "HRES_CP_FRCST_GEOOBJ",
    "HRES_DSRP_FRCST_GEOOBJ",
    "HRES_E_FRCST_GEOOBJ",
    "HRES_FDIR_FRCST_GEOOBJ",
    "HRES_HCC_FRCST_GEOOBJ",
    "HRES_I10FG_FRCST_GEOOBJ",
    "HRES_ILSPF_FRCST_GEOOBJ",
    "HRES_LCC_FRCST_GEOOBJ",
    "HRES_LSP_FRCST_GEOOBJ",
    "HRES_LSPF_FRCST_GEOOBJ",
    "HRES_MCC_FRCST_GEOOBJ",
    "HRES_MSL_FRCST_GEOOBJ",
    "HRES_PEV_FRCST_GEOOBJ",
    "HRES_PTYPE_FRCST_GEOOBJ",
    "HRES_RSN_FRCST_GEOOBJ",
    "HRES_SD_FRCST_GEOOBJ",
    "HRES_SF_FRCST_GEOOBJ",
    "HRES_SKT_FRCST_GEOOBJ",
    "HRES_SP_FRCST_GEOOBJ",
    "HRES_SSR_FRCST_GEOOBJ",
    "HRES_SSRC_FRCST_GEOOBJ",
    "HRES_SSRD_FRCST_GEOOBJ",
    "HRES_SSRDC_FRCST_GEOOBJ",
    "HRES_STR_FRCST_GEOOBJ",
    "HRES_STRC_FRCST_GEOOBJ",
    "HRES_STRDC_FRCST_GEOOBJ",
    "HRES_SUND_FRCST_GEOOBJ",
    "HRES_TCC_FRCST_GEOOBJ",
    "HRES_TCW_FRCST_GEOOBJ",
    "HRES_TCWV_FRCST_GEOOBJ",
    "HRES_PREC_FRCST_GEOOBJ",
    "HRES_TPRATE_FRCST_GEOOBJ",
    "HRES_UVB_FRCST_GEOOBJ",
    "HRES_VIS_FRCST_GEOOBJ",
    "HRES_100U_FRCST_GEOOBJ",
    "HRES_100V_FRCST_GEOOBJ",
    "HRES_100S_FRCST_GEOOBJ",
    "HRES_100A_FRCST_GEOOBJ",
    "HRES_10S_FRCST_GEOOBJ",
    "HRES_10A_FRCST_GEOOBJ",
]

# Настройки для логера
if platform == "linux" or platform == "linux2":
    logging.basicConfig(
        filename="/var/log/log-execute/log_journal_skm_ecmwf_rsv.log.txt",
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(funcName)s: %(lineno)d - %(message)s"
        ),
    )
elif platform == "win32":
    logging.basicConfig(
        filename=(
            f"{pathlib.Path(__file__).parent.absolute()}"
            "/log_journal_skm_ecmwf_rsv.log.txt"
        ),
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(funcName)s: %(lineno)d - %(message)s"
        ),
    )


# Загружаем yaml файл с настройками
with open(
    f"{pathlib.Path(__file__).parent.absolute()}/settings.yaml", "r"
) as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings["telegram"])
sql_settings = pd.DataFrame(settings["sql_db"])
pyodbc_settings = pd.DataFrame(settings["pyodbc_db"])
skm_ecmwf_settings = pd.DataFrame(settings["skm_ecmwf"])

API_KEY = skm_ecmwf_settings.api_key[0]


# Функция отправки уведомлений в telegram на любое количество каналов
# (указать данные в yaml файле настроек)
def telegram(i, text):
    try:
        msg = urllib.parse.quote(str(text))
        bot_token = str(telegram_settings.bot_token[i])
        channel_id = str(telegram_settings.channel_id[i])

        retry_strategy = Retry(
            total=3,
            status_forcelist=[101, 429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        http.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_id}&text={msg}",
            verify=False,
            timeout=10,
        )
    except Exception as err:
        print(f"skm_ecmwf: Ошибка при отправке в telegram -  {err}")
        logging.error(f"skm_ecmwf: Ошибка при отправке в telegram -  {err}")


# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    db_data = f"mysql://{user_yaml}:{password_yaml}@{host_yaml}:{port_yaml}/{database_yaml}"
    return create_engine(db_data).connect()


# Функция загрузки факта выработки
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def fact_load(i, dt):
    server = str(pyodbc_settings.host[i])
    database = str(pyodbc_settings.database[i])
    username = str(pyodbc_settings.user[i])
    password = str(pyodbc_settings.password[i])
    # Выбор драйвера в зависимости от ОС
    if platform == "linux" or platform == "linux2":
        connection_ms = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    elif platform == "win32":
        connection_ms = pyodbc.connect(
            "DRIVER={SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    #
    mssql_cursor = connection_ms.cursor()
    mssql_cursor.execute(
        "SELECT SUBSTRING (Points.PointName ,"
        "len(Points.PointName)-8, 8) as gtp, MIN(DT) as DT,"
        " SUM(Val) as Val FROM Points JOIN PointParams ON "
        "Points.ID_Point=PointParams.ID_Point JOIN PointMains"
        " ON PointParams.ID_PP=PointMains.ID_PP WHERE "
        "PointName like 'Генерация%{GVIE%' AND ID_Param=153 "
        "AND DT >= "
        + str(dt)
        + " AND PointName NOT LIKE "
        "'%GVIE0001%' AND PointName NOT LIKE '%GVIE0012%' "
        "AND PointName NOT LIKE '%GVIE0416%' AND PointName "
        "NOT LIKE '%GVIE0167%' "
        "AND PointName NOT LIKE '%GVIE0007%' "
        "AND PointName "
        "NOT LIKE '%GVIE0987%' AND PointName NOT LIKE "
        "'%GVIE0988%' AND PointName NOT LIKE '%GVIE0989%' "
        "AND PointName NOT LIKE '%GVIE0991%' AND PointName "
        "NOT LIKE '%GVIE0994%' AND PointName NOT LIKE "
        "'%GVIE1372%' GROUP BY SUBSTRING (Points.PointName "
        ",len(Points.PointName)-8, 8), DATEPART(YEAR, DT), "
        "DATEPART(MONTH, DT), DATEPART(DAY, DT), "
        "DATEPART(HOUR, DT) ORDER BY SUBSTRING "
        "(Points.PointName ,len(Points.PointName)-8, 8), "
        "DATEPART(YEAR, DT), DATEPART(MONTH, DT), "
        "DATEPART(DAY, DT), DATEPART(HOUR, DT);"
    )
    fact = mssql_cursor.fetchall()
    connection_ms.close()
    fact = pd.DataFrame(np.array(fact), columns=["gtp", "dt", "fact"])
    fact.drop_duplicates(
        subset=["gtp", "dt"], keep="last", inplace=True, ignore_index=False
    )
    fact["fact"] = fact["fact"].astype("float").round(-2)
    return fact


# Функция записи датафрейма в базу
def load_data_to_db(db_name, connect_id, dataframe):
    telegram(1, "skm_ecmwf: Старт записи в БД.")
    logging.info("skm_ecmwf: Старт записи в БД.")

    dataframe = pd.DataFrame(dataframe)
    connection_skm = connection(connect_id)
    dataframe.to_sql(
        name=db_name,
        con=connection_skm,
        if_exists="append",
        index=False,
        chunksize=5000,
    )
    rows = len(dataframe)
    telegram(1, f"skm_ecmwf: записано в БД {rows} строк ({int(rows/72)} гтп)")
    if len(dataframe.columns) > 5:
        telegram(
            0, f"skm_ecmwf: записано в БД {rows} строк ({int(rows/72)} гтп)"
        )
    logging.info(f"записано в БД {rows} строк c погодой ({int(rows/72)} гтп)")
    telegram(1, "skm_ecmwf: Финиш записи в БД.")
    logging.info("skm_ecmwf: Финиш записи в БД.")


# Функция загрузки датафрейма из базы
def load_data_from_db(
    db_name,
    col_from_database,
    connect_id,
    condition_column,
    day_interval,
):
    telegram(1, "skm_ecmwf: Старт загрузки из БД.")
    logging.info("skm_ecmwf: Старт загрузки из БД.")

    list_col_database = ",".join(col_from_database)
    connection_db = connection(connect_id)
    if day_interval is None and condition_column is None:
        query = f"select {list_col_database} from {db_name};"
    else:
        query = (
            f"select {list_col_database} from {db_name} where"
            f" {condition_column} >= CURDATE() - INTERVAL {day_interval} DAY;"
        )
    dataframe_from_db = pd.read_sql(sql=query, con=connection_db)

    telegram(1, "skm_ecmwf: Финиш загрузки из БД.")
    logging.info("skm_ecmwf: Финиш загрузки из БД.")
    return dataframe_from_db


# Раздел загрузки прогноза погоды в базу
def load_forecast_to_db(date_beg, date_end, api_key, col_parameters):
    telegram(1, "skm_ecmwf: Старт загрузки погоды.")

    ses_list = [
        [42054, "GKZ00001"],
        ...
        [42104, "GVIE0679"],
    ]

    # Загрузка прогнозов погоды по станциям
    final_dataframe = pd.DataFrame()
    g = 0

    for ses in ses_list:
        col_parameters_ses = [f"{col}_{ses[0]}" for col in col_parameters]
        list_parameters = ",".join(col_parameters_ses)

        try:
            url_response = requests.get(
                (
                    "https://exergy.skmmp.ru/api/webquery/execute?fileformat=json"
                    f"&series={list_parameters}&start={date_beg}&end={date_end}"
                    f"&interval=hour&token={api_key}&emptydata=yes&dateFormat=svse"
                    "&numberFormat=nothousandsdot&headers=yes"
                ),
                verify=False,
            )

            sleep(5)
            while url_response.status_code != 200:
                url_response = requests.get(
                    (
                        "https://exergy.skmmp.ru/api/webquery/execute?fileformat=json"
                        f"&series={list_parameters}&start={date_beg}&end={date_end}"
                        f"&interval=hour&token={api_key}&emptydata=yes&dateFormat=svse"
                        "&numberFormat=nothousandsdot&headers=yes"
                    ),
                    verify=False,
                )
                sleep(20)
            if url_response.ok:
                json_string = url_response.json()
                # print(json_string)
                weather_dataframe = pd.DataFrame(
                    data=json_string["data"],
                    columns=["datetime_msc"] + col_parameters,
                )
                weather_dataframe.insert(0, "gtp", ses[1])
                weather_dataframe.insert(
                    2, "loadtime", datetime.datetime.now().isoformat()
                )

                final_dataframe = final_dataframe.append(
                    weather_dataframe, ignore_index=True
                )

                g += 1

                print(g)
                print(ses[1])
                logging.info(
                    f"{g} Прогноз погоды для ГТП {ses[1]} загружен с skm"
                )
            else:
                print(f"skm_ecmwf: Ошибка запроса: {url_response.status_code}")
                logging.error(
                    f"skm_ecmwf: Ошибка запроса: {url_response.status_code}"
                )
                telegram(
                    1,
                    f"skm_ecmwf: Ошибка запроса: {url_response.status_code}",
                )
                # os._exit(1)
        except requests.HTTPError as http_err:
            print(f"skm_ecmwf: HTTP error occurred: {http_err.response.text}")
            logging.error(
                f"skm_ecmwf: HTTP error occurred: {http_err.response.text}"
            )
            telegram(
                1,
                f"skm_ecmwf: HTTP error occurred: {http_err.response.text}",
            )
            # os._exit(1)
        except Exception as err:
            print(f"skm_ecmwf: Other error occurred: {err}")
            logging.error(f"skm_ecmwf: Other error occurred: {err}")
            telegram(1, f"skm_ecmwf: Other error occurred: {err}")
            # os._exit(1)

    final_dataframe["datetime_msc"] = final_dataframe["datetime_msc"].astype(
        "datetime64[ns]"
    )
    final_dataframe["loadtime"] = final_dataframe["loadtime"].astype(
        "datetime64[ns]"
    )
    final_dataframe.fillna(0, inplace=True)

    gtp_dict = [
        ("GVIE0011", ["GVIE0010"]),
        ...
        ("GVIE0695", ["GVIE0689", "GVIE0691"]),
    ]
    for pair in gtp_dict:
        temp = final_dataframe[final_dataframe.gtp == pair[0]]
        for x in pair[1]:
            temp.gtp = x
            final_dataframe = pd.concat([final_dataframe, temp], axis=0)
        final_dataframe = final_dataframe.sort_values(
            by=["gtp", "datetime_msc"]
        ).reset_index(drop=True)

    final_dataframe.drop_duplicates(
        subset=["datetime_msc", "gtp"],
        keep="last",
        inplace=True,
        ignore_index=False,
    )
    final_dataframe.reset_index(drop=True, inplace=True)
    # final_dataframe.to_excel(
    #     f"{pathlib.Path(__file__).parent.absolute()}/"
    #     f"{datetime.datetime.today().strftime('%d.%m.%Y')}_weather_skm_ecmwf.xlsx"
    # )

    telegram(1, f"skm_ecmwf: загружен прогноз для {int(g)} гтп")
    logging.info(f"Сформирован датафрейм для {g} гтп")

    load_data_to_db(
        "skm_ecmwf",
        0,
        final_dataframe,
    )


# Загрузка прогнозов погоды по станциям из базы и подготовка датафреймов
def prepare_datasets_to_train():
    col_in_database = ["gtp", "datetime_msc", "loadtime"] + COL_PARAMETERS

    ses_dataframe = load_data_from_db(
        "visualcrossing.ses_gtp",
        ["gtp", "def_power"],
        0,
        None,
        None,
    )
    ses_dataframe["def_power"] = ses_dataframe["def_power"] * 1000
    # ses_dataframe = ses_dataframe[
    #     ses_dataframe["gtp"].str.contains("GVIE", regex=False)
    # ]
    ses_dataframe = ses_dataframe[
        (ses_dataframe["gtp"].str.contains("GVIE", regex=False))
        | (ses_dataframe["gtp"].str.contains("GKZ", regex=False))
        | (ses_dataframe["gtp"].str.contains("GROZ", regex=False))
    ]
    logging.info("Загружен датафрейм с гтп и установленной мощностью.")

    forecast_dataframe = load_data_from_db(
        "visualcrossing.skm_ecmwf",
        col_in_database,
        0,
        "loadtime",
        80,
    )
    logging.info("Загружен массив прогноза погоды за предыдущие дни")

    # Удаление дубликатов прогноза
    forecast_dataframe.drop_duplicates(
        subset=["datetime_msc", "gtp"],
        keep="last",
        inplace=True,
        ignore_index=False,
    )
    forecast_dataframe["month"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).month
    forecast_dataframe["hour"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).hour

    test_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"] < str(DATE_BEG_PREDICT)
            )[0]
        ]
    )
    test_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"] > str(DATE_END_PREDICT)
            )[0]
        ],
        inplace=True,
    )
    test_dataframe = test_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )

    forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"]
                > str(datetime.datetime.today())
            )[0]
        ],
        inplace=True,
    )
    # Сортировка датафрейма по гтп и дате
    forecast_dataframe.sort_values(["gtp", "datetime_msc"], inplace=True)
    forecast_dataframe["datetime_msc"] = forecast_dataframe[
        "datetime_msc"
    ].astype("datetime64[ns]")
    logging.info("forecast и test dataframe преобразованы в нужный вид")

    #
    fact = fact_load(0, "DATEADD(HOUR, -80 * 24, DATEDIFF(d, 0, GETDATE()))")

    forecast_dataframe = forecast_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )
    forecast_dataframe = forecast_dataframe.merge(
        fact,
        left_on=["gtp", "datetime_msc"],
        right_on=["gtp", "dt"],
        how="left",
    )

    forecast_dataframe.dropna(subset=["fact"], inplace=True)
    forecast_dataframe.drop(["dt", "loadtime"], axis="columns", inplace=True)

    # forecast_dataframe.to_excel("forecast_dataframe_rsv.xlsx")
    # test_dataframe.to_excel("test_dataframe_rsv.xlsx")

    for col in COL_PARAMETERS:
        forecast_dataframe[col] = forecast_dataframe[col].astype("float")
        test_dataframe[col] = test_dataframe[col].astype("float")

    col_to_int = ["month", "hour"]
    for col in col_to_int:
        forecast_dataframe[col] = forecast_dataframe[col].astype("int")
        test_dataframe[col] = test_dataframe[col].astype("int")

    logging.info("Датафреймы погоды и факта выработки склеены")
    return forecast_dataframe, test_dataframe


# Раздел подготовки прогноза на XGBoost
def prepare_forecast_xgboost(forecast_dataframe, test_dataframe):
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )

    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    # z['gtp'] = z['gtp'].str.replace('GKZV', '4')
    z["gtp"] = z["gtp"].str.replace("GKZ", "2")
    z["gtp"] = z["gtp"].str.replace("GROZ", "3")
    x = z.drop(["fact", "datetime_msc"], axis=1)
    y = z["fact"].astype("float")

    predict_dataframe = test_dataframe.drop(
        ["datetime_msc", "loadtime"], axis=1
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZV', '4')
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace("GKZ", "2")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GROZ", "3"
    )

    x["gtp"] = x["gtp"].astype("int")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")

    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.9
    )
    logging.info("Старт предикта на XGBoostRegressor")

    param = {
        "lambda": 0.5036337875040199,
        "alpha": 6.383617215973705,
        "colsample_bytree": 0.4,
        "subsample": 0.6,
        "learning_rate": 0.018,
        "n_estimators": 10000,
        "max_depth": 5,
        "random_state": 1500,
        "min_child_weight": 16,
    }
    reg = xgb.XGBRegressor(**param)
    regr = BaggingRegressor(base_estimator=reg, n_estimators=3, n_jobs=-1).fit(
        x_train, y_train
    )
    # regr = reg.fit(x_train, y_train)
    predict = regr.predict(predict_dataframe)
    test_dataframe["forecast"] = pd.DataFrame(predict)
    test_dataframe["forecast"] = test_dataframe["forecast"] * OVERVALUE_COEFF

    # test_dataframe.to_excel(
    #     f"{pathlib.Path(__file__).parent.absolute()}/test_dataframe.xlsx"
    # )

    # Важность фич, перед запуском и раскомменчиванием поменять regr выше.

    # feature_importance = reg.get_booster().get_score(importance_type='weight')
    # importance_df = pd.DataFrame()
    # importance_df['feature'] = pd.Series(feature_importance.keys())
    # importance_df['weight'] = pd.Series(feature_importance.values())
    # importance_df.sort_values(['weight'], inplace=True)
    # print(importance_df)

    logging.info("Подготовлен прогноз на XGBRegressor")
    #
    # Обработка прогнозных значений

    # Расчет исторического максимума выработки
    # для обрезки прогноза не по максимуму за месяц
    fact_historical = fact_load(0, "2015-04-01")
    fact_historical["month"] = pd.to_datetime(fact_historical.dt.values).month
    fact_historical["hour"] = pd.to_datetime(fact_historical.dt.values).hour
    gtp_dataframe = pd.DataFrame()
    for gtp in fact_historical.gtp.value_counts().index:
        gtp_month = fact_historical.loc[fact_historical.gtp == gtp]
        # print(gtp_month)
        for month in gtp_month.month.value_counts().index:
            max_month = (
                gtp_month.loc[
                    gtp_month.month == month, ["gtp", "month", "hour", "fact"]
                ]
                .groupby(["hour"], as_index=False)
                .max()
            )
            gtp_dataframe = gtp_dataframe.append(max_month, ignore_index=True)

    gtp_dataframe.sort_values(["gtp", "month"], inplace=True)
    gtp_dataframe.reset_index(drop=True, inplace=True)
    gtp_dataframe = gtp_dataframe.reindex(
        columns=["gtp", "month", "hour", "fact"]
    )
    gtp_dataframe.fact[gtp_dataframe.fact < 50] = 0

    # Расчет максимума за месяц в часы
    max_month_dataframe = pd.DataFrame()
    date_cut = (
        datetime.datetime.today() + datetime.timedelta(days=-29)
    ).strftime("%Y-%m-%d")
    cut_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(forecast_dataframe["datetime_msc"] < str(date_cut))[0]
        ]
    )
    for gtp in cut_dataframe.gtp.value_counts().index:
        max_month = (
            cut_dataframe.loc[
                cut_dataframe.gtp == gtp, ["fact", "hour", "gtp"]
            ]
            .groupby(by=["hour"])
            .max()
        )
        max_month_dataframe = max_month_dataframe.append(
            max_month, ignore_index=True
        )
    max_month_dataframe["hour"] = cut_dataframe["hour"]

    # max_month_dataframe.to_excel(
    #     f"{pathlib.Path(__file__).parent.absolute()}/max_month_dataframe.xlsx"
    # )

    # gtp_dataframe.to_excel(
    #     f"{pathlib.Path(__file__).parent.absolute()}/gtp_dataframe.xlsx"
    # )

    test_dataframe = test_dataframe.merge(
        max_month_dataframe,
        left_on=["gtp", "hour"],
        right_on=["gtp", "hour"],
        how="left",
    )
    test_dataframe = test_dataframe.merge(
        gtp_dataframe,
        left_on=["gtp", "month", "hour"],
        right_on=["gtp", "month", "hour"],
        how="left",
    )

    test_dataframe["fact"] = test_dataframe[["fact_x", "fact_y"]].max(axis=1)
    test_dataframe.drop(["fact_x", "fact_y"], axis="columns", inplace=True)

    # Если прогноз отрицательный, то 0
    test_dataframe.forecast[test_dataframe.forecast < 0] = 0

    # Интерполяция для случаев когда края с маленькой генерацией нулевые
    test_dataframe["forecast"] = np.where(
        test_dataframe["forecast"] == 0,
        (
            np.where(
                test_dataframe["fact"] > 0, np.NaN, test_dataframe.forecast
            )
        ),
        test_dataframe.forecast,
    )
    test_dataframe["forecast"].interpolate(
        method="linear", axis=0, inplace=True
    )
    #

    test_dataframe["forecast"] = test_dataframe[
        ["forecast", "fact", "def_power"]
    ].min(axis=1)

    test_dataframe.drop(
        ["fact", "month", "hour"], axis="columns", inplace=True
    )
    test_dataframe.drop(
        COL_PARAMETERS + ["loadtime", "def_power"],
        axis="columns",
        inplace=True,
    )

    # Добавить к датафрейму столбцы с текущей датой и id прогноза
    # INSERT INTO treid_03.weather_foreca (gtp,dt,id_foreca,load_time,value)
    test_dataframe.insert(2, "id_foreca", "27")
    test_dataframe.insert(3, "load_time", datetime.datetime.now().isoformat())
    test_dataframe.rename(
        columns={"datetime_msc": "dt", "forecast": "value"},
        errors="raise",
        inplace=True,
    )

    logging.info(
        f"Датафрейм с прогнозом выработки прошел обработку"
        f" от нулевых значений и обрезку по макс за месяц"
    )

    # test_dataframe.to_excel(
    #     f"{pathlib.Path(__file__).parent.absolute()}/"
    #     f"{(datetime.datetime.today() + datetime.timedelta(days=1)).strftime('%d.%m.%Y')}"
    #     "_skm_rsv.xlsx"
    # )

    # Запись прогноза в БД
    load_data_to_db("weather_foreca", 1, test_dataframe)

    # Уведомление о подготовке прогноза
    telegram(0, "skm_ecmwf: прогноз подготовлен")
    logging.info("Прогноз записан в БД treid_03")


def optuna_tune_params(forecast_dataframe, test_dataframe):
    # Подбор параметров через Optuna
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )

    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    # z['gtp'] = z['gtp'].str.replace('GKZV', '4')
    z["gtp"] = z["gtp"].str.replace("GKZ", "2")
    z["gtp"] = z["gtp"].str.replace("GROZ", "3")
    x = z.drop(["fact", "datetime_msc"], axis=1)
    y = z["fact"].astype("float")

    predict_dataframe = test_dataframe.drop(
        ["datetime_msc", "loadtime"], axis=1
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZV', '4')
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace("GKZ", "2")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GROZ", "3"
    )

    x["gtp"] = x["gtp"].astype("int")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")

    def objective(trial):
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, train_size=0.8
        )
        # 'tree_method':'gpu_hist',
        # this parameter means using the GPU when training our model
        # to speedup the training process
        param = {
            "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02],
            ),
            "n_estimators": 10000,
            "max_depth": trial.suggest_categorical(
                "max_depth", [5, 7, 9, 11, 13, 15, 17]
            ),
            "random_state": trial.suggest_categorical(
                "random_state", [500, 1000, 1500, 2000]
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        }

        reg = xgb.XGBRegressor(**param)
        reg.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=200,
        )
        prediction = reg.predict(predict_dataframe)
        score = reg.score(x_train, y_train)
        return score

    study = optuna.create_study(sampler=TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=1000, timeout=3600)
    optuna_vis = optuna.visualization.plot_param_importances(study)
    print(optuna_vis)
    print("Number of completed trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("\tBest Score: {}".format(trial.value))
    print("\tBest Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# Сам процесс работы разбит для удобства по функциям
# чтобы если погода загрузилась, а прогноз не подготовился,
#  то чтобы не тратить лимит запросов и не засорять базу,
# закомменчиваем первую функцию и разбираемся дальше сколько угодно попыток.
# 1 - load_forecast_to_db - загрузка прогноза с сайта и запись в бд
# 2 - prepare_datasets_to_train - подготовка датасетов для обучения модели,
# переменным присваиваются возвращаемые 2 датафрейма и список столбцов,
# необходимо для работы следующих функций.
# 3 - optuna_tune_params - подбор параметров для модели через оптуну
# необходимо в нее передать 2 датафрейма из предыдущей функции.
# 4 - prepare_forecast_xgboost - подготовка прогноза,
# в нее также необходимо передавать 2 датафрейма и список столбцов,
# который потом используется для удаления лишних столбцов,
# чтобы excel файл меньше места занимал.

# # 1
load_forecast_to_db(DATE_BEG, DATE_END, API_KEY, COL_PARAMETERS)
# # 2
forecast_dataframe, test_dataframe = prepare_datasets_to_train()
# # 3
# optuna_tune_params(forecast_dataframe, test_dataframe)
# # 4
prepare_forecast_xgboost(forecast_dataframe, test_dataframe)

print("Время выполнения:", datetime.datetime.now() - start_time)
