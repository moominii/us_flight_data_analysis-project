# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:28:43 2025

@author: Jiyeon Baek

USA_flight_ data.py

20250228 team project
"""
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1. 데이터 불러오기: 두 폴더의 모든 CSV 파일을 한꺼번에 읽어오기
folder1 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_1987-1999/*.csv"
folder2 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/*.csv"

# 두 폴더의 파일 리스트를 합치기
file_list = glob.glob(folder1) + glob.glob(folder2)
print("불러온 파일 개수:", len(file_list))

# 각 파일을 읽어서 리스트에 저장 (파일명에서 연도를 추출하여 'Year' 컬럼이 없으면 추가)
df_list = []
for file in file_list:
    temp_df = pd.read_csv(file)
    # 파일명에서 4자리 숫자를 찾아 연도로 사용 (예: "1987" 등)
    match = re.search(r'(\d{4})', file)
    if match:
        year = int(match.group(1))
        if 'Year' not in temp_df.columns:
            temp_df['Year'] = year
    df_list.append(temp_df)

# 모든 파일을 하나의 DataFrame으로 합치기
df_all = pd.concat(df_list, ignore_index=True)
print("전체 항공편 수:", df_all.shape[0])

# 합친 DataFrame을 CSV 파일로 저장 (지정된 경로에 저장)
output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data.csv'
df_all.to_csv(output_path, index=False)
print("CSV 파일이 저장되었습니다:", output_path)


# -------------------------------
# 2. 추가 참조 데이터 불러오기
# -------------------------------
# 각 CSV 파일은 프로젝트 폴더 내에 있다고 가정
airports_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/airports.csv")
carriers_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/carriers.csv")
plane_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/plane-data.csv")
# variable-descriptions는 분석 참고용으로 불러올 수 있음 (필요시)
var_desc_df = pd.read_csv("C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/variable-descriptions.csv")


# -------------------------------
# 3. plane-data와 df_all 병합 및 AircraftAge 계산
# -------------------------------
# plane_df의 컬럼은 모두 소문자임: 'tailnum'과 'year' 등이 존재함.
print("plane_df 컬럼:", plane_df.columns)

# 'year' 컬럼을 'PlaneYear'로 변경
plane_df.rename(columns={'year': 'PlaneYear'}, inplace=True)

# df_all의 항공기 등록번호 컬럼은 'TailNum'으로 되어 있으므로, plane_df의 'tailnum'과 merge
df_all = pd.merge(df_all, plane_df[['tailnum', 'PlaneYear']], how='left', left_on='TailNum', right_on='tailnum')

# 항공편 발생 연도(df_all['Year'])와 제조 연도(PlaneYear)를 이용해 항공기 연령 계산
df_all['AircraftAge'] = df_all['Year'] - df_all['PlaneYear']

# 불필요한 plane_df의 'tailnum' 컬럼 제거 (선택 사항)
df_all.drop(columns=['tailnum'], inplace=True)
print("병합 후 df_all의 컬럼:")
print(df_all.columns)

# -------------------------------
# 4. carriers 데이터와 병합 (항공사 코드로 친숙한 항공사명 추가)
# -------------------------------
# carriers_df에는 'Code'와 'Description' 컬럼이 있음.
df_all = pd.merge(df_all, carriers_df, how='left', left_on='UniqueCarrier', right_on='Code')
# 이제 'Description' 컬럼을 항공사명으로 활용할 수 있음.



# -------------------------------
# Q1. 최적의 시간대/요일별 지연 분석
# -------------------------------
# CRSDepTime은 HHMM 형식이므로, 시(hour) 단위로 변환
df_all['CRSDepTime'] = pd.to_numeric(df_all['CRSDepTime'], errors='coerce')
df_all['CRSDepHour'] = df_all['CRSDepTime'].apply(lambda x: int(x) // 100 if not pd.isnull(x) else np.nan)

# 요일(DayOfWeek: 1=월요일, 7=일요일)와 예약 출발 시간별 평균 출발 지연(DepDelay) 계산
delay_by_time = df_all.groupby(['DayOfWeek', 'CRSDepHour'])['DepDelay'].mean().reset_index()

# 피벗 테이블 생성 후 히트맵 시각화
delay_pivot = delay_by_time.pivot(index='DayOfWeek', columns='CRSDepHour', values='DepDelay')
plt.figure(figsize=(12,6))
sns.heatmap(delay_pivot, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("요일 및 예약 출발 시간별 평균 출발 지연 (분)")
plt.xlabel("예약 출발 시간 (시)")
plt.ylabel("요일 (1=월, 7=일)")
plt.show()

min_delay = delay_by_time['DepDelay'].min()
best_slot = delay_by_time[delay_by_time['DepDelay'] == min_delay]
print("지연이 최소인 시간대/요일:")
print(best_slot)






















