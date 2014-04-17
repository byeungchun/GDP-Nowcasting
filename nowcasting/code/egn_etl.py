#coding=utf8

# Evolutionary GDP Nowcasting
# Version 0.1
# Desc : Bloomberg, ECOS에서 입수데이터 집계
# Date : 2014. 4. 17
# Copyright :

import pandas as pd

df_bbData = pd.DataFrame()
df_bbDataCol = pd.DataFrame()

def extract_bloomberg_excel(str_bbDataFile, str_bbIndexFile):
    '''
    블룸버그에서 받아온 엑셀파일을 dataframe 형식으로 변경하여 저장
    :param str_bbDataFile: 실제 데이터 파일
    :param str_bbIndexFile: 메타파일
    '''
    
    global df_bbData, df_bbDataCol
    
    #데이터
    df_bbData = pd.read_excel(str_bbDataFile,'Sheet1')
    df_bbData = df_bbData.ix[5:,:] #제목행 및 날짜 없는행 제거
    df_bbData = df_bbData.replace('#N/A N/A','') #엑셀에서 데이터 없는 셀에 들어간 문자열 제거
    df_bbData = df_bbData.convert_objects(convert_numeric=True) #모든 컬럼은 숫자형식으로 변환
    
    #리스트
    df_bbIndex = pd.read_excel(str_bbIndexFile, 'index')
    df_bbIndex.columns = ['no','idx','cat','rgn','rgn2','rmk','undf']
    df_bbDataCol = df_bbIndex[df_bbIndex['no'].isin(df_bbData.columns)][['no','idx','rgn2']]
    
    
def extract_national_df(lst_nation):
    '''
    국가별로 데이터를 추출
    :param lst_nation: 국가, 공통변수 명 ex)['한국', '글로벌']
    '''
    
    global df_bbDataCol
    
    df_bbData1 = df_bbData.ffill() #첫행부터 데이터가 없는 개수를 파악하기 위해 중간에 빈 값을 채워줌
    df_bbData1 = df_bbData1.ix[:-2,:] #데이터가 2013년 4월 7일 이휴는 정확하지 않아 잘라줌
    df_bbDataZero = pd.DataFrame([df_bbData1.columns,map(lambda x: df_bbData1[x].dropna().count(), df_bbData1.columns)]).T
    df_bbDataZero.columns = ['idx','nozero_cnt']
    #데이터 인덱스의 이름을 가지고와서 'no','idx','nozero_cnt' 로 구성
    df_bbDataZero=pd.merge(df_bbDataCol[['no','idx']],df_bbDataZero,left_on='no',right_on='idx')
    df_bbDataZero=df_bbDataZero.ix[:,[0,1,3]]
    #실제 데이터가 있는 값이 3000개 이상인 index만 남겨놓음
    df_bbDataCol = df_bbDataCol[df_bbDataCol['no'].isin(df_bbDataZero[df_bbDataZero['nozero_cnt'] > 3000]['no'])]
    df_nation=df_bbDataCol[df_bbDataCol['rgn2'].isin(lst_nation)]
    
    return df_nation


def agg_mmQqWw(df_nation,dt_from ,dt_to):
    '''
일별 데이터를 분기, 월, 주 단위로 집계
    :param df_nation: 대상 지역 
    :param dt_from: 시작일자 ex) pd.datetime(2002,3,1)
    :param dt_to: 끝 일자 ex) pd.datetime(2014,3,31)
    '''
    
    df_daily = df_bbData[df_nation['no']]
    df_daily = df_daily.dropna()
    df_daily = df_daily.ix[dt_from:dt_to]
    df_daily = df_daily.ffill() #중간에 비어있는 셀은 직전 값을 채워 넣음
    df_dataCol = df_daily.columns.astype(int).astype(str)
    df_daily.columns = df_dataCol
    lst_how = ['mean','std','first','last','min','median','max']
    lst_cols = []
    for x1 in lst_how:
      for x2 in df_dataCol:
        lst_cols.append(x2+'_'+x1)        
    df_month = df_daily.resample('M',how=lst_how)
    df_month.columns = lst_cols
    df_quarter = df_daily.resample('Q',how=lst_how)
    df_quarter.columns = lst_cols
    df_week = df_daily.resample('W',how=lst_how)
    df_week.columns = lst_cols
    #주단위 경우, 데이터의 시작일과 끝이 주 중간에서 시작하고 끝날 수 있기 때문에 집계가 안될 수 있어 첫주와 마지막주는 fill로 매꾸어줌
    df_week=df_week.ffill()
    df_week=df_week.bfill()
    
    return df_quarter, df_month, df_week, df_daily
    






