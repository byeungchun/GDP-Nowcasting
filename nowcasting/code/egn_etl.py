#coding=utf8

# Evolutionary GDP Nowcasting
# Version 0.1
# Desc : Bloomberg, ECOS에서 입수데이터 집계
# Date : 2014. 4. 16
# Copyright :

import pandas as pd

df_bbData
df_bbDataCol

def extract_bloomberg_excel(str_bbDataFile, str_bbIndexFile):
    '''
    블룸버그에서 받아온 엑셀파일을 dataframe 형식으로 변경하여 저장
    :param str_bbDataFile: 실제 데이터 파일
    :param str_bbIndexFile: 메타파일
    '''
    
    global df_bbData, df_bbDataCol
    
    #데이터
    df_bbData = pd.read_excel(str_bbDataFile,'Sheet1')
    df_bbData = df_xls.ix[5:,:] #제목행 및 날짜 없는행 제거
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
    
    df_bbData1 = df_bbData.ffill() #첫행부터 데이터가 없는 개수를 파악하기 위해 중간에 빈 값을 채워줌
    df_bbData1 = df_bbData1.ix[:-2,:] #데이터가 2013년 4월 7일 이휴는 정확하지 않아 잘라줌
    df_bbDataZero = pd.DataFrame([df_bbData1.columns,map(lambda x: df_bbData1[x].dropna().count(), df_bbData1.columns)]).T
    df_bbDataZero.columns = ['idx','nozero_cnt']
    #데이터 인덱스의 이름을 가지고와서 'no','idx','nozero_cnt' 로 구성
    df_bbDataZero=pd.merge(df_index[['no','idx']],df_bbDataZero,left_on='no',right_on='idx')
    df_bbDataZero=df_bbDataZero.ix[:,[0,1,3]]
    #실제 데이터가 있는 값이 3000개 이상인 index만 남겨놓음
    df_bbDataCol[df_bbDataCol['no'].isin(df_bbDataZero[df_bbDataZero['nozero_cnt'] > 3000]['no'])]

    df_nation=df_bbDataCol[df_bbDataCol['rgn2'].isin(lst_nation)]
    
    return df_nation






