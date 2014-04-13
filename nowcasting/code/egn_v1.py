#coding=utf8

# Evolutionary GDP Nowcasting
# Version 0.1
# Desc : 여러 국가에 대한 Nowcasting, SVM 기법 사용
# Date : 2014. 4. 13
# Copyright :

import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
import logging
import time
import scipy.stats as ss
import os
from sklearn import svm

arr_pool = np.array([])  #모든 경제지표가 들어가는 Pool
arr_pop = np.array([])  #Pool에서 선택된 지표군
df_quarterlyData = pd.DataFrame();  #분기 집계된 데이터
df_monthlyData = pd.DataFrame();  #월단위 집계된 데이터
df_weeklyData = pd.DataFrame();  #주단위 집계된 데이터
lst_quarterlyXs = []  #분기 단위에서 사용할 경제지표 리스트
lst_quarterlyYs = []  #분기 단위 GDP
int_startingExpNum = 10000000  #실험 일련번호
lst_nowcastingResult = []  #결과저장

#사용자 파라미터
int_numChromesInPop = 100  #세대당 염색체 개수
int_probGeneLive = 100  #유전자를 고를때 살아남을 확률. 예를들어 10이면 10%, 100이면 1%, 2이면 50%

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(funcName)s][%(lineno)d] %(message)s')
np.random.seed(5252)


def init_egn():
    """
    (1)집계된 데이터 입력, (2)GDP, 지표 구분, (3)지표 Pool 생성

    """
    global arr_pool, df_quarterlyData, lst_quarterlyXs, lst_quarterlyYs, df_monthlyData, df_weeklyData

    df_quarterlyData = pd.read_csv('../data/resultQuarter2_1.csv', sep='\t')
    df_monthlyData = pd.read_csv('../data/resultMonthly2.csv', sep='\t')
    df_weeklyData = pd.read_csv('../data/resultWeekly2.csv', sep='\t')
    df_monthlyData.index = pd.to_datetime(df_monthlyData['date'])
    df_weeklyData.index = pd.to_datetime(df_weeklyData['date'])
    lst_quarterlyXs = df_quarterlyData.columns[2:]
    lst_quarterlyYs = df_quarterlyData.columns[1]

    for idx in lst_quarterlyXs:
        arr_pool = np.append(arr_pool, idx)


def evaluate_chrome(df_quarterlyResize, str_exprPeriod, int_populationNo):
    """
    (1)염색체별 SVM fitting (2)평가 (3)결과저장
    :param df_quarterlyResize: 분기 데이터
    :param str_exprPeriod: 실험분기
    :param int_populationNo: 세대 번호
    :return: 세대 평가점수
    """
    global arr_pop, int_startingExpNum, lst_nowcastingResult

    clf = svm.SVR()
    lst_chromeEvaluationResult = []
    dbl_evaluationWeight = 1  # 평가함수 가중치
    arr_yTrainingValues = np.array(df_quarterlyResize[lst_quarterlyYs])[:-1]
    arr_yTesting1Values = np.array(df_quarterlyResize[lst_quarterlyYs])[-2:-1]
    arr_yTesting2Values = np.array(df_quarterlyResize[lst_quarterlyYs])[-4:]

    #세대에서 염색체를 뽑아 SVM fitting을 하고 평가하여 결과를 저장
    for arr_genes in arr_pop:
        lst_genes = arr_genes[:]
        logging.debug(str(int_startingExpNum) + '번호 염색체:' + str(lst_genes))
        arr_xValues= sm.add_constant(np.array(df_quarterlyResize[lst_genes]), prepend=True) #X변수에 상수 1을 포함
        arr_xTesting1Values = arr_xValues[-2:-1]    #t-1 시점
        arr_xTesting2Values = arr_xValues[-4:]      #t-4 ~ t 시점
        arr_xTrainingValues = arr_xValues[:-1]      #1 ~ t-1 시점
        arr_xMonthly = sm.add_constant(np.array(df_monthlyData[lst_genes]), prepend=True)  #월별 GDP 예측용 값
        arr_xWeekly = sm.add_constant(np.array(df_weeklyData[lst_genes]), prepend=True)  #주별 GDP 예측용 값
        
        clf.fit(arr_xTrainingValues, arr_yTrainingValues)
        
        lst_chromeResult = []
        lst_genes.append(lst_quarterlyYs)
        dbl_trainingPeriodCoef = np.corrcoef(arr_yTrainingValues, clf.predict(arr_xTrainingValues))[0, 1] #훈련기간 실측-예측 상관계수
        arr_predictValuesTesting1Quarterly = clf.predict(arr_xTesting1Values)
        arr_predictValuesTesting2Quarterly = clf.predict(arr_xTesting2Values)
        arr_predictValuesMonthly = clf.predict(arr_xMonthly)
        arr_predictValuesWeekly = clf.predict(arr_xWeekly)
        dbl_testingPeriodCoef = np.corrcoef(arr_yTesting2Values, arr_predictValuesTesting2Quarterly)[0, 1]

        if np.isnan(dbl_testingPeriodCoef): dbl_testingPeriodCoef = 0.0 #상관계수가 계산이 안되면 0으로 표시
        dbl_predictionErrorCdf = float(ss.norm(0, 1).cdf(abs(arr_yTesting1Values - arr_predictValuesTesting1Quarterly)))

        if np.isnan(dbl_trainingPeriodCoef): #상관계수가 계산이 안되면 -9999.0으로 표시
            dbl_trainingPeriodCoef = -9999.0
        dbl_evaluationValue = calc_evaluation(dbl_trainingPeriodCoef, dbl_predictionErrorCdf, dbl_evaluationWeight)
        lst_chromeEvaluationResult.append(dbl_evaluationValue)  #평가점수

        lst_chromeResult.append(str_exprPeriod)     #기간
        lst_chromeResult.append(int_startingExpNum) #일련번호
        lst_chromeResult.append(int_populationNo)   #세대번호
        lst_chromeResult.append(arr_genes)          #지표
        lst_chromeResult.append(dbl_trainingPeriodCoef) #상관계수
        lst_chromeResult.append(dbl_predictionErrorCdf) #t-1 예측오차
        lst_chromeResult.append(dbl_evaluationWeight)   #가중치
        lst_chromeResult.append(dbl_evaluationValue)    #평가점수
        lst_chromeResult.append(arr_yTesting2Values[-1])    #T시점 GDP
        lst_chromeResult.append(arr_predictValuesTesting2Quarterly[-1]) #T시점예측GDP
        lst_chromeResult.append(arr_yTesting2Values)    #테스트기간 실측GDP
        lst_chromeResult.append(arr_predictValuesTesting2Quarterly) #테스트기간 예측GDP
        lst_chromeResult.append(str(arr_predictValuesMonthly.tolist())[1:-1])   #테스트기간 예측GDP(월단위)
        lst_chromeResult.append(str(arr_predictValuesWeekly.tolist())[1:-1])    #테스트기간 예측GDP(주단위)

        lst_nowcastingResult.append(lst_chromeResult)

        int_startingExpNum += 1

    return lst_chromeEvaluationResult


def calc_evaluation(dbl_rSquare, dbl_predictionErrorCdf, dbl_evaluationWeight):
    """
    염색체 평가함수(상관계수 및 t-1 시점 예측오차에 대한 가중평균)
    :param dbl_rSquare: 훈련기간 예측-실측 상관계수
    :param dbl_predictionErrorCdf: t-1 시점의 예측오차
    :param dbl_evaluationWeight: 가중치
    :return: 염색체 평가점수
    """
    dbl_rvsMeanVal = (1.0 - dbl_predictionErrorCdf) * 2  #CDF는 0.5가 가장 좋고 커질수록 나빠지기 때문에 조정
    return dbl_rSquare * dbl_evaluationWeight + dbl_rvsMeanVal * (1.0 - dbl_evaluationWeight)


def add_chrome_to_population(chrome):
    """
    염색체를 세대에 삽입하기 위한 함수
    :param chrome: 염색체
    """
    global arr_pop

    lst_temp1 = [] if len(arr_pop) == 0 else arr_pop.tolist()
    if ~(chrome.tolist() in arr_pop.tolist()) & len(chrome) != 0:  #이미 arr_pop에 해당 크롬이 있거나 크롬의 크기가 0일 경우 건너 뛰게금 설정
        lst_temp1.append(chrome.tolist())
        arr_pop = np.array(lst_temp1)


def generate_population(int_populationIndex):
    """
    세대 생성(첫번째 세대인 경우에만 모든 염색체, 그 다음부터는 부족한 염색체만 생성
    :param int_populationIndex: 세대번호
    """
    global arr_pool, arr_pop, int_numChromesInPop, int_probGeneLive
    if int_populationIndex == 0: arr_pop = np.array([])  #첫번째 염색체인 경우에만 초기화
    #염색체 개수를 생성
    while len(arr_pop) != int_numChromesInPop:
        chrome = arr_pool[~np.random.randint(int_probGeneLive, size=len(arr_pool)).astype(bool)]
        add_chrome_to_population(chrome)
    logging.debug(str(int_populationIndex) + '세대 염색체 수:' + str(len(arr_pop)))


def evolve_population(df_populationTable, int_populationIndex):
    """
진화 프로세스
    :param df_populationTable: 기존 세대 현황판
    :param int_populationIndex: 세대번호
    """
    global arr_pool, arr_pop, int_numChromesInPop, int_probGeneLive
    arr_pop = np.array([])

    #하위 염색체 탈락
    df_populationTable = df_populationTable.ix[df_populationTable.sort(columns=['evaluation'], ascending=False).index[:-int(len(df_populationTable) * 0.4)]]
    logging.debug(str(int_populationIndex) + '세대 잔여 염색체 수:' + str(len(df_populationTable)))

    #상위 염색체 존속
    for i in range(int(int_numChromesInPop * 0.2)):
        add_chrome_to_population(np.array(df_populationTable.ix[df_populationTable.index[i]][0]))
    logging.debug('상위 10% 존속 개수:' + str(len(arr_pop)))

    #교배 및 염색체 신규생성을 통해 작업
    arr_popIdx = df_populationTable.index
    while len(arr_popIdx) > 1:
        arr_pickIdx = random.sample(arr_popIdx, 2)
        #새로운 염색체를 만듬
        arr_newChrome = np.array([])
        for i in arr_pickIdx:
            #logging.debug('교배를 위해 선택된 염색체:'+str(df_pop.ix[i].tolist()))
            for j in df_populationTable.ix[i][0]:
                if np.random.random() > 0.5 and j not in arr_newChrome:  #1/2확률로 추가
                    arr_newChrome = np.append(arr_newChrome, j)
        add_chrome_to_population(arr_newChrome)
        logging.debug(str(int_populationIndex) + '세대 추가 후보 염색체:' + str(pd.DataFrame(arr_newChrome)[0].str[-3:].tolist()))
        arr_popIdx = arr_popIdx - arr_pickIdx
    if len(arr_popIdx) == 1:
        logging.debug('교배되지 못한 염색체:' + str(df_populationTable.ix[arr_popIdx[0]][0]))
        add_chrome_to_population(np.array(df_populationTable.ix[arr_popIdx[0]][0]))
    logging.debug(str(int_populationIndex) + '세대 염색체 수(교배후): ' + str(len(arr_pop)))


def execute_egn():
    global lst_nowcastingResult, date_forecastingPeriod, df_monthlyData, df_weeklyData

    int_gap = 21 #훈련기간(20) + 예측기간(1)
    init_egn()
    arr_testingPoints = np.arange(0, len(df_quarterlyData), 1)
    for int_testingPoint in arr_testingPoints:  #기간에 따른 조건 변경
        if int_testingPoint == 1: break  #마지막 값에는 GDP가 없기 때문에 계산 안함
        if (int_testingPoint + int_gap) > len(df_quarterlyData): continue
        df_quarterlyResize = df_quarterlyData[int_testingPoint:int_testingPoint + int_gap]
        date_forecastingPeriod = pd.to_datetime(df_quarterlyResize['date'][-2:])
        df_monthlyData = df_monthlyData.ix[date_forecastingPeriod.values[1]:][1:]
        df_weeklyData = df_weeklyData.ix[date_forecastingPeriod.values[1]:][1:]
        logging.info(str(int_testingPoint) + '분기, GDP:' + str(list(df_quarterlyResize['gdp'])))

        for int_populationIndex in range(100):  # 세대수
            if int_populationIndex == 0:
                generate_population(0) #첫번째 세대
                continue
            arr_populationEvaluation = evaluate_chrome(df_quarterlyResize, str(int_testingPoint) + '_' + str(int_gap), int_populationIndex) #세대 결과값
            df_populationTable = pd.DataFrame(arr_pop, columns=['chrome']); #세대 현황
            df_populationTable['evaluation'] = pd.DataFrame(arr_populationEvaluation)
            evolve_population(df_populationTable, int_populationIndex)    #다음세대로의 진화
            generate_population(int_populationIndex)  #새로운 세대에서 부족한 염색체 생성

    df_finalResult = pd.DataFrame(lst_nowcastingResult)
    df_finalResult.columns = ['기간', '일련번호', '세대번호', '지표', '상관계수', 't-1 예측오차', '가중치', '평가점수',
                              'T시점 GDP', 'T시점예측GDP','테스트기간 실측GDP', '테스트기간 예측GDP', '테스트기간 예측GDP(월)', '테스트기간 예측GDP(주)']
    df_finalResult.to_csv('../output/egn_v1_' + time.strftime("%m%d%H%M") + '.csv', sep='\t', index=False)


if __name__ == '__main__':
    os.chdir(os.getcwd())
    execute_egn()
