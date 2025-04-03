# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:45:16 2025

@author: Jiyeon Baek

flight_data_code.py
"""

import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from matplotlib import font_manager, rc 
import platform 


if platform.system() == 'Windows':
    path = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname = path).get_name()
    rc('font', family = font_name)
elif platform.system() == 'Darwin':
    rc('font', family = 'AppleGothic')
else:
    print('Check your OS system')

# ===============================
# 1. 데이터 병합 및 원본 CSV 파일 저장
# ===============================
# 두 폴더(1987-1999, 2000-2008)의 CSV 파일 경로 (절대 경로 사용)
folder1 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_1987-1999/*.csv"
folder2 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/*.csv"


# 두 폴더의 파일 목록을 합침
file_list = glob.glob(folder1) + glob.glob(folder2)
print("불러온 파일 개수:", len(file_list))

# 각 파일을 읽어 DataFrame 리스트에 저장 (파일명에서 연도 추출 후 'Year' 컬럼 추가)
df_list = []
for file in file_list:
    try:
        temp_df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        # UTF-8로 읽기 실패 시 cp949로 시도
        temp_df = pd.read_csv(file, encoding='cp949')
    match = re.search(r'(\d{4})', file)
    if match:
        year = int(match.group(1))
        if 'Year' not in temp_df.columns:
            temp_df['Year'] = year
    df_list.append(temp_df)

# 모든 파일을 하나의 DataFrame으로 합침
df_all = pd.concat(df_list, ignore_index=True)
print("전체 항공편 수:", df_all.shape[0])

# 병합한 원본 데이터를 CSV 파일로 저장
output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data.csv'
df_all.to_csv(output_path, index=False)
print("병합된 원본 CSV 파일이 저장되었습니다:", output_path)


# ===============================
# 2. 불필요한 컬럼 제거 후 축소 CSV 파일 저장
# ===============================
# 분석 주제(Q1~Q8)에 크게 필요하지 않은 컬럼들 (예: 실제 시각, 비행번호, 운항 시간 관련 세부정보 등)
cols_to_drop = [
    'DepTime',         # 실제 출발 시각 (예약 시각 사용)
    'ArrTime',         # 실제 도착 시각
    'CRSArrTime',      # 예약 도착 시각
    'FlightNum',       # 항공편 번호
    'CRSElapsedTime',  # 예약 운항 시간
    'AirTime',         # 실제 공중 비행 시간
    'TaxiIn',          # 착륙 후 택시 시간
    'TaxiOut',         # 이륙 전 택시 시간
    'CancellationCode',# 취소 사유 (분석 목적에 따라 생략)
    'Diverted',        # 우회 여부
    'NASDelay',        # 항공 관제 관련 지연
    'SecurityDelay'    # 보안 관련 지연
]

# 불필요한 컬럼 삭제
df_reduced = df_all.drop(columns=cols_to_drop)
print("축소된 데이터프레임 컬럼:", df_reduced.columns)

# 축소된 데이터를 새 CSV 파일로 저장
reduced_output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data_reduced.csv'
df_reduced.to_csv(reduced_output_path, index=False)
print("불필요한 컬럼을 제거한 CSV 파일이 저장되었습니다:", reduced_output_path)


# ===============================
# 3. 추가 참조 데이터 병합 (plane-data, carriers)
# ===============================
# 참조 데이터 파일들은 작업 디렉토리에 있다고 가정
airports_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/airports.csv")
carriers_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/carriers.csv")
plane_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/plane-data.csv")
var_desc_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/variable-descriptions.csv")  # 필요 시 참고

# plane-data 파일의 컬럼은 모두 소문자: 'tailnum', 'year', ... 
print("plane_df 컬럼:", plane_df.columns)

# 'year' 컬럼을 'PlaneYear'로 변경
plane_df.rename(columns={'year': 'PlaneYear'}, inplace=True)

# df_reduced의 항공기 등록번호 컬럼은 'TailNum'임. plane_df의 'tailnum'과 병합
df_reduced = pd.merge(df_reduced, plane_df[['tailnum', 'PlaneYear']], how='left',
                      left_on='TailNum', right_on='tailnum')

# 항공편 발생 연도(Year)와 제조 연도(PlaneYear)로 항공기 연식(AircraftAge) 계산
df_reduced['AircraftAge'] = df_reduced['Year'] - df_reduced['PlaneYear']

# 불필요한 병합 후의 'tailnum' 컬럼 제거
df_reduced.drop(columns=['tailnum'], inplace=True)
print("plane-data 병합 후 df_reduced 컬럼:", df_reduced.columns)

# carriers 데이터와 병합하여 항공사명(Description) 추가 (carriers_df의 'Code' 기준)
df_reduced = pd.merge(df_reduced, carriers_df, how='left',
                      left_on='UniqueCarrier', right_on='Code')
# 이제 'Description' 컬럼으로 항공사명을 활용 가능


































