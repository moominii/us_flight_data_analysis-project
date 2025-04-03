# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:57:38 2025

@author: Jiyeon Baek

data_analy_flight.py


"""

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# ------------------------------
# 1. 파일 병합 및 원본 CSV 저장
# ------------------------------
# 두 폴더의 절대 경로 (파일명은 "YYYY.csv" 형태여야 함)
folder1 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_1987-1999/*.csv"
folder2 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/*.csv"

# 두 폴더의 모든 파일 목록
file_list = glob.glob(folder1) + glob.glob(folder2)

# "YYYY.csv" 패턴에 맞는 파일만 선택 (참조 파일 제외)
flight_files = [f for f in file_list if re.search(r'[\\/]\d{4}\.csv$', f)]
print("Flight files:")
for f in flight_files:
    print(f)


# 각 파일을 읽어와 리스트에 저장 (파일명에서 연도를 추출하여 'Year' 컬럼 추가)
df_list = []
for file in flight_files:
    base = os.path.basename(file)  # 예: "1987.csv"
    match = re.search(r'(\d{4})', base)
    if match:
        year = int(match.group(1))
    else:
        continue  # 연도 추출 실패 시 건너뜀

    success = False
    for encoding in ['utf-8', 'cp949', 'latin1']:
        try:
            # low_memory 옵션 제거 (engine='python' 사용 시 지원되지 않음)
            temp_df = pd.read_csv(file, encoding=encoding, engine='python')
            print(f"파일 {file}을(를) {encoding} 인코딩으로 성공적으로 읽었습니다.")
            success = True
            break
        except Exception as e:
            print(f"{encoding} 인코딩 실패: {file} -> {e}")
    if not success:
        print("모든 인코딩 시도 실패하여 파일 건너뜁니다:", file)
        continue

    if 'Year' not in temp_df.columns:
        temp_df['Year'] = year
    df_list.append(temp_df)


import dask.dataframe as dd

# Dask를 사용하여 모든 파일을 읽어오면, 메모리 제한 문제를 피할 수 있습니다.
df_dask = dd.read_csv(flight_files, encoding='latin1')  # 이미 인코딩 문제는 해결된 상태라 가정
print("Dask DataFrame shape (지연 평가):", df_dask.shape)
# 필요한 작업을 수행하고, 최종 결과만 메모리에 로드합니다.
df_all = df_dask.compute()  # compute() 시 실제 메모리에 로드함
print("Merged flight data shape:", df_all.shape)


# 원본 데이터 CSV 저장
original_output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data3.csv'
df_all.to_csv(original_output_path, index=False)
print("원본 CSV 파일이 저장되었습니다:", original_output_path)


# ------------------------------
# 2. 불필요한 컬럼 제거 및 축소 CSV 저장
# ------------------------------
# 분석 주제에 필수적이지 않은 컬럼들 삭제
cols_to_drop = [
    'DepTime',         # 실제 출발 시각 (예약 시각 사용)
    'ArrTime',         # 실제 도착 시각
    'CRSArrTime',      # 예약 도착 시각
    'FlightNum',       # 항공편 번호
    'CRSElapsedTime',  # 예약 운항 시간
    'AirTime',         # 실제 비행 시간
    'TaxiIn',          # 착륙 후 택시 시간
    'TaxiOut',         # 이륙 전 택시 시간
    'CancellationCode',# 취소 사유 코드 (분석 목적에 따라 생략)
    'Diverted',        # 우회 여부
    'NASDelay',        # 항공 관제 지연
    'SecurityDelay'    # 보안 관련 지연
]

df_reduced = df_all.drop(columns=cols_to_drop)
print("축소된 데이터프레임 컬럼:", df_reduced.columns)

# 축소된 데이터 CSV 저장
reduced_output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data_reduced2.csv'
df_reduced.to_csv(reduced_output_path, index=False)
print("축소된 CSV 파일이 저장되었습니다:", reduced_output_path)

























