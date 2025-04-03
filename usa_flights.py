# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:32:19 2025

@author: Jiyeon Baek


usa_flights.py
"""

import os
import re
import glob
import pandas as pd
import numpy as np
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
    
    
    

# 메모리 최적화를 위한 함수 (수치형 컬럼 다운캐스팅)
def optimize_dataframe(df):
    # float64 → float32로 다운캐스팅
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    # int64 → int32로 다운캐스팅
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df


# ------------------------------
# 1. 항공편 데이터 파일 병합 및 원본 CSV 저장
# ------------------------------
# 두 폴더 경로 (참조 파일 제외)
folder1 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_1987-1999/*.csv"
folder2 = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/*.csv"

# 두 폴더의 모든 파일 목록
file_list = glob.glob(folder1) + glob.glob(folder2)

# "YYYY.csv" 형태의 파일들만 선택 (airports, carriers, plane-data, variable-descriptions 제외)
flight_files = [f for f in file_list if re.search(r'[\\/]\d{4}\.csv$', f)]
print("Flight files:")
for f in flight_files:
    print(f)


# 각 파일을 읽어와서 DataFrame 리스트에 저장 (파일명에서 연도 추출 후 'Year' 강제 업데이트)
df_list = []
for file in flight_files:
    base = os.path.basename(file)  # 예: "1987.csv"
    match = re.search(r'(\d{4})', base)
    if match:
        year = int(match.group(1))
    else:
        continue  # 연도 추출 실패 시 건너뜁니다.
    
    success = False
    # 인코딩 순서: utf-8 → cp949 → latin1 (errors='replace' 사용)
    for encoding in ['utf-8', 'cp949', 'latin1']:
        try:
            with open(file, 'r', encoding=encoding, errors='replace') as f:
                temp_df = pd.read_csv(f, engine='python')
            print(f"파일 {file}을(를) {encoding} 인코딩(대체 모드)로 성공적으로 읽었습니다.")
            success = True
            break
        except Exception as e:
            print(f"{encoding} 인코딩 (대체 모드) 실패: {file} -> {e}")
    if not success:
        print("모든 인코딩 시도 실패하여 파일 건너뜁니다:", file)
        continue
    
    # 무조건 파일명에서 추출한 연도로 'Year' 컬럼 업데이트
    temp_df['Year'] = year
    
    # 메모리 최적화: 수치형 컬럼 다운캐스팅
    temp_df = optimize_dataframe(temp_df)
    
    df_list.append(temp_df)

# 파일들을 하나의 DataFrame으로 병합 (다운캐스팅 덕분에 메모리 사용량 감소)
df_all = pd.concat(df_list, ignore_index=True)
print("Merged flight data shape:", df_all.shape)

# 병합된 원본 데이터를 CSV 파일로 저장
original_output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data5.csv'
df_all.to_csv(original_output_path, index=False)
print("병합된 원본 CSV 파일이 저장되었습니다:", original_output_path)


df_all.head(3)
'''
   Year  Month  DayofMonth  ...  NASDelay  SecurityDelay  LateAircraftDelay
0  1987     10          14  ...       NaN            NaN                NaN
1  1987     10          15  ...       NaN            NaN                NaN
2  1987     10          17  ...       NaN            NaN     
'''



df_all.tail(3)
'''
   Year  Month  DayofMonth  ...  NASDelay  SecurityDelay  LateAircraftDelay
118914455  2008      4          17  ...       NaN            NaN                NaN
118914456  2008      4          17  ...       NaN            NaN                NaN
118914457  2008      4          17  ...       NaN            NaN                NaN
'''


# ------------------------------
# 2. 불필요한 컬럼 제거 및 축소 CSV 저장
# ------------------------------
cols_to_drop = [
    'DepTime', 'ArrTime', 'CRSArrTime', 'FlightNum', 'CRSElapsedTime',
    'AirTime', 'TaxiIn', 'TaxiOut', 'CancellationCode', 'Diverted',
    'NASDelay', 'SecurityDelay'
]

df_reduced = df_all.drop(columns=cols_to_drop)
print("축소된 데이터프레임 컬럼:", df_reduced.columns)

reduced_output_path = r'C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data_reduced5.csv'
df_reduced.to_csv(reduced_output_path, index=False)
print("축소된 CSV 파일이 저장되었습니다:", reduced_output_path)

df_reduced = pd.read_csv('C:/Users/Admin/Desktop/JY/LG U+/20250228_project/all_data_reduced5.csv')

df_reduced.head(3)
'''
   Year  Month  DayofMonth  ...  CarrierDelay  WeatherDelay LateAircraftDelay
0  1987     10          14  ...           NaN           NaN               NaN
1  1987     10          15  ...           NaN           NaN               NaN
2  1987     10          17  ...           NaN           NaN               NaN
'''


df_reduced.tail(3)
'''
 Year  Month  ...  WeatherDelay  LateAircraftDelay
118914455  2008      4  ...           NaN                NaN
118914456  2008      4  ...           NaN                NaN
118914457  2008      4  ...           NaN                NaN
'''



# ------------------------------
# 3. 참조 데이터 병합 (plane-data, carriers, airports)
# ------------------------------
# 참조 파일 경로 (필요시 수정)
airports_path = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/airports.csv"
carriers_path = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/carriers.csv"
plane_data_path = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/plane-data.csv"
var_desc_path = r"C:/Users/Admin/Desktop/JY/LG U+/20250228_project/dataverse_files_2000-2008/variable-descriptions.csv"

airports_df = pd.read_csv(airports_path, encoding='utf-8')
carriers_df = pd.read_csv(carriers_path, encoding='utf-8')
plane_df = pd.read_csv(plane_data_path, encoding='utf-8')
var_desc_df = pd.read_csv(var_desc_path, encoding='utf-8')  # 필요시 참고

# plane_df의 'year' 컬럼을 'PlaneYear'로 변경 (plane_df의 컬럼은 모두 소문자)
plane_df.rename(columns={'year': 'PlaneYear'}, inplace=True)

# df_reduced와 plane_df를 'TailNum'과 'tailnum' 기준으로 병합 (왼쪽 병합)
df_reduced = pd.merge(df_reduced, plane_df[['tailnum', 'PlaneYear']], how='left',
                      left_on='TailNum', right_on='tailnum')
df_reduced['AircraftAge'] = df_reduced['Year'] - df_reduced['PlaneYear']
df_reduced.drop(columns=['tailnum'], inplace=True)

# carriers 데이터와 병합해 항공사명을 추가 ('UniqueCarrier'와 'Code' 기준)
df_reduced = pd.merge(df_reduced, carriers_df, how='left',
                      left_on='UniqueCarrier', right_on='Code')

# 음수 AircraftAge 제거 (잘못된 데이터) 또는 AircraftAge가 null인 경우는 그대로 둠
df_reduced = df_reduced[df_reduced['AircraftAge'].isna() | (df_reduced['AircraftAge'] >= 0)]
print("참조 데이터 병합 후 df_reduced의 컬럼:")
print(df_reduced.columns)

df_reduced.describe() 
'''
df_reduced.describe() 
Out[37]: 
               Year         Month  ...     PlaneYear   AircraftAge
count  1.188817e+08  1.188817e+08  ...  3.989625e+07  3.989625e+07
mean   1.998260e+03  6.483119e+00  ...  1.992639e+03  1.046374e+01
std    6.061115e+00  3.462488e+00  ...  5.036366e+01  5.030763e+01
min    1.987000e+03  1.000000e+00  ...  0.000000e+00  0.000000e+00
25%    1.993000e+03  3.000000e+00  ...  1.989000e+03  4.000000e+00
50%    1.999000e+03  6.000000e+00  ...  1.994000e+03  8.000000e+00
75%    2.004000e+03  1.000000e+01  ...  2.000000e+03  1.300000e+01
max    2.008000e+03  1.200000e+01  ...  2.008000e+03  2.008000e+03
'''

# ------------------------------
# 4. 데이터 분석 (Q1 ~ Q8)
# ------------------------------
# =============================================================================
# Q1. 출발 지연 최소화를 위한 최적 시간대/요일 분석
# =============================================================================

flight_data = df_reduced
# 월 평균 출발 지연
monthly_avg_delay = df_reduced.groupby("Month")["DepDelay"].mean().reset_index()

# 선 그래프 시각화
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_delay, x="Month", y="DepDelay", marker="o", color="b")
plt.xlabel("Month")
plt.ylabel("Average Departure Delay (minutes)")
plt.title("Monthly Average Departure Delay (1995-2008)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# 시간대 & 요일별 출발 지연 평균 계산
df_reduced["Hour"] = df_reduced["CRSDepTime"] // 100  # 시간(HH만 추출)
heatmap_data = df_reduced.pivot_table(index="DayOfWeek", columns="Hour", values="DepDelay", aggfunc="mean")

# 9월 데이터 필터링
september_data = df_reduced[df_reduced["Month"] == 9]
september_data["Hour"] = september_data["CRSDepTime"] // 100  # 시간(HH만 추출)
heatmap_september = september_data.pivot_table(index="DayOfWeek", columns="Hour", values="DepDelay", aggfunc="mean")

# 9월 히트맵 시각화
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_september, cmap="coolwarm", annot=True, linewidths=0.5)
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week (1=Mon, 7=Sun)")
plt.title("Average Departure Delay in September by Time and Day of Week")
plt.show()

# 12월 데이터 필터링
december_data = df_reduced[df_reduced["Month"] == 12]
december_data["Hour"] = december_data["CRSDepTime"] // 100  # 시간(HH만 추출)
heatmap_december = december_data.pivot_table(index="DayOfWeek", columns="Hour", values="DepDelay", aggfunc="mean")

# 12월 히트맵 시각화
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_december, cmap="coolwarm", annot=True, linewidths=0.5)
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week (1=Mon, 7=Sun)")
plt.title("Average Departure Delay in December by Time and Day of Week")
plt.show()


# =============================================================================
# Q2. 오래된 비행기일수록 지연이 더 잦은지? (도착 지연 기준)
# =============================================================================
# 비행기 연식별 평균 도착 지연 시간 계산
age_arrival_delay = df_reduced.groupby("AircraftAge")["ArrDelay"].mean().reset_index()

age_arrival_delay = age_arrival_delay[age_arrival_delay['AircraftAge'] >= 0]
age_arrival_delay = age_arrival_delay[age_arrival_delay['AircraftAge'] < 60]

# 그래프 설정
sns.lineplot(x="AircraftAge", y="ArrDelay", data=age_arrival_delay, marker='o', color='r')
plt.title("Average Arrival Delay by Aircraft Age")
plt.xlabel("Aircraft Age (Years)")
plt.ylabel("Average Arrival Delay (Minutes)")

plt.tight_layout()
plt.show()


# =============================================================================
# Q3. 시간이 지남에 따라 다양한 장소 간을 비행하는 사람의 수 변화
# =============================================================================
# 연도별 항공편 수 계산
yearly_flights = df_reduced.groupby("Year").size().reset_index(name="FlightCount")

# 월별 항공편 수 계산
monthly_flights = df_reduced.groupby(["Year", "Month"]).size().reset_index(name="FlightCount")

# 그래프 그리기
plt.figure(figsize=(12, 5))

# 연도별 항공편 수 트렌드
plt.subplot(1, 2, 1)
sns.lineplot(data=yearly_flights, x="Year", y="FlightCount", marker='o')
plt.title("Total Flights per Year")
plt.xlabel("Year")
plt.ylabel("Number of Flights")

# 월별 항공편 수 변화 트렌드
plt.subplot(1, 2, 2)
sns.lineplot(data=monthly_flights, x="Month", y="FlightCount", hue="Year", palette="coolwarm")
plt.title("Monthly Flight Trends Over the Years")
plt.xlabel("Month")
plt.ylabel("Number of Flights")

plt.tight_layout()
plt.show()



# =============================================================================
# Q4. 날씨가 비행기 지연을 얼마나 예측하는지?
# =============================================================================
df_reduced["WeatherImpact_Dep"] = df_reduced["WeatherDelay"] / df_reduced["DepDelay"]
df_reduced["WeatherImpact_Arr"] = df_reduced["WeatherDelay"] / df_reduced["ArrDelay"]

# 2. 상관관계 분석
correlation = df_reduced[["WeatherDelay", "DepDelay", "ArrDelay"]].corr()
print("Correlation Matrix:\n", correlation)

# 3. 월별 날씨 지연 패턴 분석
monthly_weather_delay = df_reduced.groupby("Month")["WeatherDelay"].mean()

# 그래프 그리기
plt.figure(figsize=(12, 5))

# 월별 날씨 지연 평균
plt.subplot(1, 2, 1)
sns.lineplot(x=monthly_weather_delay.index, y=monthly_weather_delay.values, marker='o')
plt.title("Average Weather Delay by Month")
plt.xlabel("Month")
plt.ylabel("Average Weather Delay (minutes)")

# 히트맵으로 상관관계 시각화
plt.subplot(1, 2, 2)
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Weather Delay & Flight Delays")

plt.tight_layout()
plt.show()






# =============================================================================
# Q5. 한 공항의 지연이 다른 항공편의 지연에 미치는 연쇄적 실패 (체인 리액션) 분석
# =============================================================================
import networkx as nx
# 기존 데이터에서 평균 출발 및 도착 지연 계산
delay_network = df_reduced.groupby(["Origin", "Dest"])[["DepDelay", "ArrDelay"]].mean().reset_index()

# 1. 평균 지연이 30분 이상인 공항만 필터링
high_delay_routes = delay_network[(delay_network["DepDelay"] > 30) & (delay_network["ArrDelay"] > 30)]

# 2. 그래프 생성
G = nx.DiGraph()

# 3. 필터링된 데이터 기반으로 노드 및 엣지 추가
for _, row in high_delay_routes.iterrows():
    G.add_edge(row["Origin"], row["Dest"], weight=row["DepDelay"])

# 4. Degree 기준으로 필터링 (연결된 공항이 5개 이상인 경우만)
node_degree = dict(G.degree())
filtered_nodes = [node for node, degree in node_degree.items() if degree >= 5]
G_filtered = G.subgraph(filtered_nodes)

# 5. 네트워크 그래프 시각화
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_filtered, k=0.5)  # 노드 간격 조정

# 노드 크기를 연결된 개수(degree)에 따라 가변적으로 설정
node_sizes = [node_degree[node] * 200 for node in G_filtered.nodes()]

nx.draw(G_filtered, pos, with_labels=True, node_size=node_sizes, font_size=10, edge_color="red")
plt.title("Simplified Flight Delay Propagation Network")
plt.show()








# =============================================================================
# Q6. 9/11 이전과 이후의 비행 패턴 비교
# =============================================================================
# 연도별 비행 횟수 계산
flights_per_year = df_reduced.groupby("Year").size()

# 연도별 평균 지연 시간 계산
average_delay_per_year = df_reduced.groupby("Year")["ArrDelay"].mean()

# 시각화 설정
fig, ax1 = plt.subplots(figsize=(10,5))

# 첫 번째 그래프 (비행 횟수 변화)
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Flights', color=color)
ax1.plot(flights_per_year.index, flights_per_year.values, marker='o', color=color, label='Number of Flights')
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 그래프 (평균 지연 시간 변화)
ax2 = ax1.twinx()  # Y축 공유
color = 'tab:red'
ax2.set_ylabel('Average Delay (minutes)', color=color)
ax2.plot(average_delay_per_year.index, average_delay_per_year.values, marker='s', linestyle='--', color=color, label='Avg Delay Time')
ax2.tick_params(axis='y', labelcolor=color)

# 제목 및 범례 추가
fig.suptitle('Yearly Flight Count and Average Delay Time')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 그래프 출력
plt.show()







# =============================================================================
# Q7. 가장 자주 운항하는 두 도시 간 항공편 비교
# =============================================================================
# 2007년 데이터 필터링
flight_data_2007 = df_reduced[df_reduced["Year"] == 2007]

# 가장 자주 운항된 두 도시 찾기
route_counts = flight_data_2007.groupby(["Origin", "Dest"]).size().reset_index(name="count")
top_route = route_counts.nlargest(1, "count")

# 해당 구간의 데이터 필터링
top_origin, top_dest = top_route.iloc[0]["Origin"], top_route.iloc[0]["Dest"]
filtered_df = flight_data_2007[(flight_data_2007["Origin"] == top_origin) & (flight_data_2007["Dest"] == top_dest)]

# 이상치 제거 (지연 시간이 180분 이하인 데이터만 사용)
filtered_df = filtered_df[(filtered_df["DepDelay"].between(0, 120)) & (filtered_df["ArrDelay"].between(0, 120))]

# 시각화 - 출발 및 도착 지연 비교
plt.figure(figsize=(12, 6))
sns.histplot(filtered_df["DepDelay"], bins=30, color="blue", label="Departure Delay", kde=True)
sns.histplot(filtered_df["ArrDelay"], bins=30, color="orange", label="Arrival Delay", kde=True)
plt.legend()
plt.title(f"Flight Delay Distribution (2007): {top_origin} ↔ {top_dest}")
plt.xlabel("Delay (minutes)")
plt.ylabel("Count")
plt.show()







# =============================================================================
# Q8. 시카고(ORD) 관련 항공편 비교
# =============================================================================
# ORD(시카고 오헤어 공항)와 가장 많이 연결된 상위 5개 경로


# ORD를 포함하는 항공편 필터링
ord_flights = df_reduced[(df_reduced["Origin"] == "ORD") | (df_reduced["Dest"] == "ORD")]

# ORD와 가장 많이 연결된 공항 찾기
ord_routes = ord_flights.groupby(["Origin", "Dest"]).size().reset_index(name="count")
top_ord_route = ord_routes.nlargest(5, "count")  # 가장 많은 항공편이 운항된 경로 5개 선택

# 상위 5개 공항의 평균 출발 지연 & 도착 지연 비교
top_5_airports = top_ord_route[["Origin", "Dest"]].values.flatten()
filtered_top_5_df = ord_flights[(ord_flights["Origin"].isin(top_5_airports)) & 
                                (ord_flights["Dest"].isin(top_5_airports))]

delay_means = filtered_top_5_df.groupby(["Origin", "Dest"])[["DepDelay", "ArrDelay"]].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=delay_means.melt(id_vars=["Origin", "Dest"], var_name="DelayType", value_name="Minutes"),
            x="Origin", y="Minutes", hue="DelayType", palette={"DepDelay": "blue", "ArrDelay": "orange"})

plt.title("Average Departure & Arrival Delay for Top 5 Routes")
plt.xlabel("Airport")
plt.ylabel("Average Delay (minutes)")
plt.legend(title="Delay Type")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# 상위 5개 공항의 시간대별 평균 지연 비교
filtered_top_5_df["Hour"] = (filtered_top_5_df["CRSDepTime"] // 100).astype(int)  # 시간대 추출
hourly_delay_top_5 = filtered_top_5_df.groupby(["Hour", "Origin"])[["DepDelay", "ArrDelay"]].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_delay_top_5, x="Hour", y="DepDelay", hue="Origin", marker="o", palette="tab10")
plt.title("Average Departure Delay by Hour for Top 5 Airports")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Departure Delay (minutes)")
plt.xticks(range(0, 24))
plt.legend(title="Airport")
plt.grid(True)
plt.show()



# ----------------------------------------------------------------
# 기존 ORD 관련 상위 5개 경로 데이터 필터링 (이미 전처리된 df_reduced 사용)
# ----------------------------------------------------------------
# ORD를 포함하는 항공편 필터링
ord_flights = df_reduced[(df_reduced["Origin"] == "ORD") | (df_reduced["Dest"] == "ORD")]

# ORD와 연결된 경로별 항공편 수 계산
ord_routes = ord_flights.groupby(["Origin", "Dest"]).size().reset_index(name="count")
# 상위 5개 경로 선택 (항공편 수 기준)
top_ord_route = ord_routes.nlargest(5, "count")

# 상위 5개 경로에 속하는 공항 코드들을 추출 (양방향을 모두 포함)
top_5_airports = top_ord_route[["Origin", "Dest"]].values.flatten()
top_5_airports = np.unique(top_5_airports)  # 중복 제거

# ORD를 포함한 항공편 중, 출발지와 도착지가 상위 5개 공항에 해당하는 데이터 필터링
filtered_top5_df = ord_flights[(ord_flights["Origin"].isin(top_5_airports)) & 
                                (ord_flights["Dest"].isin(top_5_airports))]
print("상위 5개 경로에 해당하는 데이터 행 수:", filtered_top5_df.shape[0])

# ----------------------------------------------------------------
# Q9. 상위 5개 경로에 대해 비행거리와 항공편 지연(DepDelay, ArrDelay) 관계 분석
# ----------------------------------------------------------------
# 분석에 사용할 변수: Distance, DepDelay, ArrDelay
df_top5_analysis = filtered_top5_df[['Distance', 'DepDelay', 'ArrDelay']].dropna()

# --- 1. 비행거리와 출발 지연(DepDelay)의 관계 ---
plt.figure(figsize=(10,6))
sns.scatterplot(x='Distance', y='DepDelay', data=df_top5_analysis, alpha=0.3)
plt.title("상위 5개 경로: 비행거리 vs 출발 지연 (DepDelay)")
plt.xlabel("비행거리 (마일)")
plt.ylabel("출발 지연 (분)")
plt.show()

# 선형 회귀 분석: Distance -> DepDelay
X_dep = df_top5_analysis[['Distance']]
y_dep = df_top5_analysis['DepDelay']
model_dep = LinearRegression()
model_dep.fit(X_dep, y_dep)
y_pred_dep = model_dep.predict(X_dep)
r2_dep = r2_score(y_dep, y_pred_dep)
print(f"상위 5개 경로 - 비행거리와 출발 지연(DepDelay) 간 R-squared: {r2_dep:.4f}")

plt.figure(figsize=(10,6))
sns.scatterplot(x='Distance', y='DepDelay', data=df_top5_analysis, alpha=0.3)
plt.plot(df_top5_analysis['Distance'], y_pred_dep, color='red', linewidth=2)
plt.title("상위 5개 경로: 비행거리 vs 출발 지연 (DepDelay) - 회귀선 포함")
plt.xlabel("비행거리 (마일)")
plt.ylabel("출발 지연 (분)")
plt.show()

# --- 2. 비행거리와 도착 지연(ArrDelay)의 관계 ---
plt.figure(figsize=(10,6))
sns.scatterplot(x='Distance', y='ArrDelay', data=df_top5_analysis, alpha=0.3)
plt.title("상위 5개 경로: 비행거리 vs 도착 지연 (ArrDelay)")
plt.xlabel("비행거리 (마일)")
plt.ylabel("도착 지연 (분)")
plt.show()

# 선형 회귀 분석: Distance -> ArrDelay
X_arr = df_top5_analysis[['Distance']]
y_arr = df_top5_analysis['ArrDelay']
model_arr = LinearRegression()
model_arr.fit(X_arr, y_arr)
y_pred_arr = model_arr.predict(X_arr)
r2_arr = r2_score(y_arr, y_pred_arr)
print(f"상위 5개 경로 - 비행거리와 도착 지연(ArrDelay) 간 R-squared: {r2_arr:.4f}")

plt.figure(figsize=(10,6))
sns.scatterplot(x='Distance', y='ArrDelay', data=df_top5_analysis, alpha=0.3)
plt.plot(df_top5_analysis['Distance'], y_pred_arr, color='red', linewidth=2)
plt.title("상위 5개 경로: 비행거리 vs 도착 지연 (ArrDelay) - 회귀선 포함")
plt.xlabel("비행거리 (마일)")
plt.ylabel("도착 지연 (분)")
plt.show()

# --- 3. 분석 데이터 요약 통계 출력 ---
print("상위 5개 경로 분석 데이터 요약 통계:")
print(df_top5_analysis.describe())


# 1. Distance 구간화 (예: 300~400, 400~500, 500~600, 600~700, 700~800)
bins = [300, 400, 500, 600, 700, 800]
labels = ["300-400", "400-500", "500-600", "600-700", "700-800"]

# Distance를 구간화하여 새로운 컬럼 DistanceBin 생성
df_top5_analysis['DistanceBin'] = pd.cut(df_top5_analysis['Distance'], bins=bins, labels=labels, include_lowest=True)

# 2. 구간별 평균 출발 지연(DepDelay)과 도착 지연(ArrDelay) 계산
bin_delay = df_top5_analysis.groupby('DistanceBin')[['DepDelay', 'ArrDelay']].mean().reset_index()

# 3. Melt하여 한 그래프에 출발/도착 지연을 모두 표현
bin_delay_melt = bin_delay.melt(id_vars='DistanceBin', 
                                value_vars=['DepDelay','ArrDelay'], 
                                var_name='DelayType', 
                                value_name='MeanDelay')

# 4. 막대그래프로 시각화
plt.figure(figsize=(10,6))
sns.barplot(data=bin_delay_melt, x='DistanceBin', y='MeanDelay', hue='DelayType', palette='viridis')
plt.title("상위 5개 ORD 경로 - 거리 구간별 평균 출발/도착 지연")
plt.xlabel("비행거리 구간 (마일)")
plt.ylabel("평균 지연 (분)")
plt.legend(title="지연 유형")
plt.show()

# 5. 구간별 데이터 개수(항공편 수)도 확인 (필요시)
bin_count = df_top5_analysis.groupby('DistanceBin').size().reset_index(name='FlightCount')
print("거리 구간별 항공편 수:")
print(bin_count)




