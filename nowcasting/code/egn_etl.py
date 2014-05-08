#coding=utf8

# Evolutionary GDP Nowcasting
# Version 0.1
# Desc : Bloomberg, ECOS에서 입수데이터 집계
# Date : 2014. 4. 17
# Copyright :

import pandas as pd
import numpy as np
import sqlite3 as lite
from pandas.io import sql

df_bbData = pd.DataFrame()
df_bbDataCol = pd.DataFrame()

def extract_bloomberg_excel(str_bbDataFile, str_bbIndexFile,is_excel):
    '''
    블룸버그에서 받아온 엑셀파일을 dataframe 형식으로 변경하여 저장
    :param str_bbDataFile: 실제 데이터 파일
    :param str_bbIndexFile: 메타파일
    '''
    
    global df_bbData, df_bbDataCol
    
    if(is_excel):
        #데이터
        df_bbData = pd.read_excel(str_bbDataFile,'Sheet1')
        df_bbData = df_bbData.ix[5:,:] #제목행 및 날짜 없는행 제거
        df_bbData = df_bbData.replace('#N/A N/A','') #엑셀에서 데이터 없는 셀에 들어간 문자열 제거
        df_bbData = df_bbData.convert_objects(convert_numeric=True) #모든 컬럼은 숫자형식으로 변환
        
        #리스트
        df_bbIndex = pd.read_excel(str_bbIndexFile, 'index')
        df_bbIndex.columns = ['no','idx','cat','rgn','rgn2','rmk','undf']
        df_bbDataCol = df_bbIndex[df_bbIndex['no'].isin(df_bbData.columns)][['no','idx','rgn2']]
        
        #csv로 저장
        df_bbData.to_csv('../data/DailyEconomicData.csv',sep='\t',encoding='utf-8')
        df_bbDataCol.to_csv('../data/index.csv',sep='\t',encoding='utf-8')
    else:
        df_bbData = pd.read_csv('../data/DailyEconomicData.csv',sep='\t',encoding='utf-8')
        df_bbDataCol = pd.read_csv('../data/index.csv',sep='\t',encoding='utf-8')


def join_ecos_bloomberg(str_ecosXlsFile, str_bbCsvFile):
    '''
    ecos 데이터와  bloomberg 데이터를 합침
    :param str_ecosXlsFile:
    :param str_bbCsvFile:
    '''
    
    global df_bbData
    
    bb = pd.read_csv(str_bbCsvFile,sep='\t', encoding='utf-8')
    bb = bb[:5209] #제일 끝 2개의 행에 날짜 형식이 아닌 값이 들어 있어 제거
    bb.index = pd.to_datetime(bb[bb.columns[0]],format='%Y-%m-%d')
    bb = bb[bb.columns[1:]]
    
    ecos = pd.read_excel(str_ecosXlsFile,'data')
    ecos.index = pd.to_datetime(ecos['date'].astype(str),format='%Y%m%d')
    ecos = ecos[ecos.columns[1:]]
    ecos.columns=map(lambda x: x[-3:], ecos.columns)
    bb=bb['20000102':'20131231'] #Bloomberg의 데이터 양이 더 적기 때문에 bb를 기준으로 자름. 다만 bb에 날짜형식이 아닌 데이터가 포함되어 있기 때문에 제거함
    
    df_bbData = bb.join(ecos)
    
    #셀의 값이 0인 것 중에 컬럼평균이 10보다 크면 그 값을 null로 변환
    for i in df_bbData.columns:
      if df_bbData[i].mean > 10:
        df_bbData[i][df_bbData[i] == 0] = np.nan

    
    
def extract_national_df(lst_nation):
    '''
    국가별로 데이터를 추출
    :param lst_nation: 국가, 공통변수 명 ex)['한국', '글로벌']
    '''
    
    global df_bbDataCol
    
    df_bbData1 = df_bbData.ffill() #첫행부터 데이터가 없는 개수를 파악하기 위해 중간에 빈 값을 채워줌
    #df_bbData1 = df_bbData1.ix[:-2,:] #데이터가 2013년 4월 7일 이휴는 정확하지 않아 잘라줌
    df_bbDataZero = pd.DataFrame([df_bbData1.columns,map(lambda x: df_bbData1[x].dropna().count(), df_bbData1.columns)]).T
    df_bbDataZero.columns = ['no','nozero_cnt']
    #데이터 인덱스의 이름을 가지고와서 'no','idx','nozero_cnt' 로 구성
    #df_bbDataZero=pd.merge(df_bbDataCol[['no','idx']],df_bbDataZero,on='no')
    df_bbDataCol = df_bbDataCol[['no','idx','rgn2']]
    df_bbDataCol['nozero_cnt'] = df_bbDataZero['nozero_cnt']
    #df_bbDataZero=df_bbDataZero.ix[:,[0,1,3]]
    #실제 데이터가 있는 값이 3000개 이상인 index만 남겨놓음
    df_bbDataCol = df_bbDataCol[df_bbDataCol['nozero_cnt'] > 3000]
    #df_bbDataCol = df_bbDataCol[df_bbDataCol['no'].isin(df_bbDataZero[df_bbDataZero['nozero_cnt'] > 3000]['no'])]
    df_nation=df_bbDataCol[df_bbDataCol['rgn2'].isin(lst_nation)]
    
    return df_nation


def agg_mmQqWw(df_nation,dt_from ,dt_to):
    '''
일별 데이터를 분기, 월, 주 단위로 집계
    :param df_nation: 대상 지역 
    :param dt_from: 시작일자 ex) pd.datetime(2002,3,1)
    :param dt_to: 끝 일자 ex) pd.datetime(2014,3,31)
    '''
    
    df_daily = df_bbData[df_nation['no'].astype(str)]
    df_daily = df_daily.dropna()
    df_daily = df_daily.ix[dt_from:dt_to]
    df_daily = df_daily.ffill() #중간에 비어있는 셀은 직전 값을 채워 넣음
    df_dataCol = df_daily.columns.astype(int).astype(str)
    df_daily.columns = df_dataCol
    lst_how = ['mean','var','first','last','min','median','max']
    lst_cols = []
    for x1 in lst_how:
      for x2 in df_dataCol:
        lst_cols.append(x2+'_'+x1)        
    df_month = df_daily.resample('M',how=lst_how)
    df_month.columns = lst_cols
    df_quarter = df_daily.resample('Q',how=lst_how)
    df_quarter.columns = lst_cols
    df_quarter['date'] = df_quarter.index
    df_week = df_daily.resample('W',how=lst_how)
    df_week.columns = lst_cols
    #주단위 경우, 데이터의 시작일과 끝이 주 중간에서 시작하고 끝날 수 있기 때문에 집계가 안될 수 있어 첫주와 마지막주는 fill로 매꾸어줌
    df_week=df_week.ffill()
    df_week=df_week.bfill()
    
    return df_quarter, df_month, df_week, df_daily
  
  
def agg_mmQqWw2(dt_from ,dt_to):
    '''
일별 데이터를 분기, 월, 주 단위로 집계
    :param dt_from: 시작일자 ex) pd.datetime(2002,3,1)
    :param dt_to: 끝 일자 ex) pd.datetime(2014,3,31)
    '''
    df_daily = df_bbData
    #df_daily = df_daily.dropna()
    df_daily = df_daily.ix[dt_from:dt_to]
    df_daily = df_daily.ffill() #중간에 비어있는 셀은 직전 값을 채워 넣음
    df_daily = df_daily.bfill() #데이터가 처음부터 없는 경우는 뒤어 값을 채워 넣음 (추후에는 없으면 빼는 것으로 2014.5.8)
    df_dataCol = df_daily.columns
    #df_daily.columns = df_dataCol
    lst_how = ['mean','var','first','last','min','median','max']
    lst_cols = []
    for x1 in lst_how:
      for x2 in df_dataCol:
        lst_cols.append(x2+'_'+x1)        
    df_month = df_daily.resample('M',how=lst_how)
    df_month.columns = lst_cols
    df_quarter = df_daily.resample('Q',how=lst_how)
    df_quarter.columns = lst_cols
    df_quarter['date'] = df_quarter.index
    df_week = df_daily.resample('W',how=lst_how)
    df_week.columns = lst_cols
    #주단위 경우, 데이터의 시작일과 끝이 주 중간에서 시작하고 끝날 수 있기 때문에 집계가 안될 수 있어 첫주와 마지막주는 fill로 매꾸어줌
    df_week=df_week.ffill()
    df_week=df_week.bfill()
    
    return df_quarter, df_month, df_week, df_daily



def extract_gdp_excel(xlsx_file, sheet_name):
    df_gdp = pd.read_excel(xlsx_file,sheet_name)
    #Ecos_gdp.xlsx의 시작은 2000년 1분기
    df_gdp.index =pd.date_range('20000101',periods=df_gdp.shape[0],freq='Q')
    
    return df_gdp


def request_data_from_processing(is_excel,lst_nation,dt_from,dt_to):
    '''
GUI에서 데이터와 일자를 일괄적으로 요첳할때 호출하는 함수  
    :param is_excel: 엑셀파일에서 직접 혹은 엑셀을 정리한 csv 에서 할 것인지
    :param lst_nation:
    :param dt_from:
    :param dt_to:
    '''
    
    global df_bbData, df_bbDataCol
    
    if(is_excel):
        extract_bloomberg_excel('../data/DailyEconomicData.xlsx', '../data/index.xlsx',True)
    else:
        extract_bloomberg_excel('../data/DailyEconomicData.csv','../data/index.csv', False)
        df_bbData.index=pd.to_datetime(df_bbData[df_bbData.columns[0]])
        df_bbData = df_bbData[df_bbData.columns[1:]]
        df_bbData.index.name = 'date'
        #df_bbDataCol.index = df_bbDataCol[df_bbDataCol.columns[0]]
        df_bbDataCol = df_bbDataCol[df_bbDataCol.columns[1:]]

    #bloomberg와 ecos를 합쳐서 보여줌
    join_ecos_bloomberg('../data/ecos_daily.xls','../data/DailyEconomicData.csv')

    df_nation = extract_national_df(lst_nation)
    df_quarter,df_month,df_week,df_daily = agg_mmQqWw(df_nation,dt_from,dt_to)
    
    df_gdp = extract_gdp_excel('../data/Ecos_gdp.xlsx','Sheet1')
    df_gdp = df_gdp.ix[df_quarter.index] #df_quarter가 가지고 있는 범위 만큼만 잘라줌
    df_gdp = df_gdp[lst_nation[0]] #첫번째가 국가, 두번째는 글로벌이기 때문
    
    df_quarter['gdp'] = df_gdp
    
    return df_gdp, df_quarter,df_month,df_week,df_daily



def request_data_from_db(nations,dt_from,dt_to):
  
    global df_bbData
    #bloomberg와 ecos를 합쳐서 보여줌
    con = lite.connect('../data/nowcasting.db')
    df_idxData = sql.read_frame('select * from idx_data',con=con)
    df_idxIndex = sql.read_frame('select * from idx_desc',con=con)
    df_gdp = sql.read_frame('select * from idx_gdp',con=con)
    
    df_idxData.index = pd.to_datetime(df_idxData[df_idxData.columns[0]])
    df_idxData = df_idxData[df_idxData.columns[1:]]
    
    df_gdp.index = pd.to_datetime(df_gdp['date'])
    df_gdp = df_gdp[df_gdp.columns[:-1]]
    
    lst_degIdx = df_idxIndex[df_idxIndex['rgn2'].isin(nations)]['num']
    df_idxData = df_idxData[df_idxData.columns[df_idxData.columns.isin(lst_degIdx)]]
    
    df_bbData = df_idxData

    #df_nation = extract_national_df(lst_nation)
    df_quarter,df_month,df_week,df_daily = agg_mmQqWw2(dt_from,dt_to)
    
    #df_gdp = extract_gdp_excel('../data/Ecos_gdp.xlsx','Sheet1')
    df_gdp = df_gdp.ix[df_quarter.index] #df_quarter가 가지고 있는 범위 만큼만 잘라줌
    df_gdp = df_gdp[nations[0].encode('utf-8')] #첫번째가 국가, 두번째는 글로벌이기 때문
    
    df_quarter['gdp'] = df_gdp
    
    return df_quarter,df_month,df_week


if __name__ == '__main__':
    #df_gdp, df_quarter,df_month,df_week,df_daily = request_data_from_processing(False,[u'한국',u'글로벌'],pd.datetime(2002,3,1),pd.datetime(2013,12,31))
    #df_gdp, df_quarter,df_month,df_week,df_daily = request_data_from_join([u'한국',u'글로벌'],pd.datetime(2002,3,1),pd.datetime(2013,12,31))
    df_quarter,df_month,df_week = request_data_from_db([u'한국',u'글로벌'],pd.datetime(2002,3,1),pd.datetime(2013,12,31))



