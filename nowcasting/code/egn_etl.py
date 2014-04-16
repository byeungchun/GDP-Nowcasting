#coding=utf8

# Evolutionary GDP Nowcasting
# Version 0.1
# Desc : Bloomberg, ECOS에서 입수데이터 집계
# Date : 2014. 4. 16
# Copyright :

import pandas as pd

def extract_bloomberg_excel(str_bbDataFile, str_bbIndexFile):
    #데이터
    df_bbData = pd.read_excel(str_bbDataFile,'Sheet1')
    df_bbData = df_xls.ix[5:,:] #제목행 및 날짜 없는행 제거
    df_bbData = df_bbData.replace('#N/A N/A','') #엑셀에서 데이터 없는 셀에 들어간 문자열 제거
    df_bbData = df_bbData.convert_objects(convert_numeric=True) #모든 컬럼은 숫자형식으로 변환
    #리스트
    df_bbIndex = pd.read_excel(str_bbIndexFile, 'index')



