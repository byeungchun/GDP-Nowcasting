#coding=utf8

######################################################
# OLS를 이용해 예측 값에 대한 적합도를 이용한 GA실행
######################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
import logging
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import svm
import time
import scipy.stats as ss
import os

arr_pool=np.array([])
arr_pop=np.array([])
int_uniqNum = 10000000
int_numChromesInPop=100 #염색체 수
int_probGeneLive=100   #유전자를 고를때 살아남을 확률. 예를들어  10이면 10%, 100이면 1%, 2이면 50%
df_stat=pd.DataFrame(); dfm=pd.DataFrame(); dfw=pd.DataFrame();
idx_x=[]
idx_y=[]
period_q=[]
np.random.seed(5252)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(funcName)s][%(lineno)d] %(message)s')
df_resFile=pd.DataFrame()
lst_resFile = []
logging.info('GA Trainig loaded')


def fnc_init():
  global arr_pool,df_stat, idx_x,idx_y,dfm,dfw
  df_stat = pd.read_csv('../data/resultQuarter2_1.csv',sep='\t')
  dfm =  pd.read_csv('../data/resultMonthly2.csv',sep='\t')
  dfw =  pd.read_csv('../data/resultWeekly2.csv',sep='\t')
  dfm.index = pd.to_datetime(dfm['date'])
  dfw.index = pd.to_datetime(dfw['date'])
  idx_x = df_stat.columns[2:]
  idx_y = df_stat.columns[1]
  for idx in idx_x:
    arr_pool = np.append(arr_pool,idx)


#염색체를 세대에 삽입하기 위한 함수
def fnc_addChromeToPop(chrome):
  global arr_pop
  lst_temp1 = [] if len(arr_pop) == 0 else arr_pop.tolist()
  #이미 arr_pop에 해당 크롬이 있거나 크롬의 크기가 0일 경우 건너 뛰게금 설정
  if ~(chrome.tolist() in arr_pop.tolist()) & len(chrome) !=0:
    lst_temp1.append(chrome.tolist())
    arr_pop=np.array(lst_temp1)
    
def fnc_evaluateChrome(df_statResize,dateFrTo,int_popNum):
  global arr_pop,df_resFile,int_uniqNum, lst_resFile
  clf=svm.SVR()
  arr_yVal = np.array(df_statResize[idx_y])[:-1]
  #logging.info('gdp'+str(list(arr_yVal)))
  arr_yTest = np.array(df_statResize[idx_y])[-2:-1]
  arr_yTest2= np.array(df_statResize[idx_y])[-4:]
  arr_chromeEval = []
  lst_score=[1000,600,300,200,150,140,130,120,110,100,50,-1000]
  dbl_calcweight = 1 #1을 주면 fitting한 값만 평가하는 것임
  for arr_genes in arr_pop:
    arr_genesTp = arr_genes[:]
    logging.debug(str(int_uniqNum)+'번호 염색체:'+str(arr_genesTp))
    arr_xVals = np.array(df_statResize[arr_genesTp])
    arr_xVals = sm.add_constant(arr_xVals,prepend=True)   
    arr_xTest = arr_xVals[-2:-1]
    arr_xTest2= arr_xVals[-4:]
    arr_xVals = arr_xVals[:-1]
    arr_xMonths = sm.add_constant(np.array(dfm[arr_genesTp]),prepend=True) #월별 GDP 예측을 위한 값
    arr_xWeeks = sm.add_constant(np.array(dfw[arr_genesTp]),prepend=True)
    #ols_res = sm.OLS(arr_yVal, arr_xVals).fit()
    clf.fit(arr_xVals,arr_yVal)
#     df_res = pd.concat((pd.DataFrame(arr_xVals[:,1:]),pd.DataFrame(arr_yVal)),1)
    lst_res = []
    arr_genesTp.append(idx_y) 
#     df_res.columns = arr_genesTp
    dbl_rSquare=np.corrcoef(arr_yVal,clf.predict(arr_xVals))[0,1]
#     df_res['fitted']=clf.predict(arr_xVals)
#     df_res['istrain'] = 'Y'
#     arr_columns = df_res.columns
    dbl_yHat=clf.predict(arr_xTest)    
    arr_yHat=clf.predict(arr_xTest2)
    arr_yHatMonth = clf.predict(arr_xMonths)
    arr_yHatWeek = clf.predict(arr_xWeeks)
    dbl_predCoef=np.corrcoef(arr_yTest2,arr_yHat)[0,1]
    if np.isnan(dbl_predCoef): dbl_predCoef=0.0
    #df_temp=pd.concat((pd.DataFrame(arr_xTest[:,1:]),pd.DataFrame(arr_yTest),pd.DataFrame(dbl_yHat)),1)
    #pdb.set_trace()
    #df_temp=pd.concat((pd.DataFrame(arr_xTest[:,1:]),pd.DataFrame(arr_yTest),pd.DataFrame(dbl_yHat),pd.DataFrame(['N'])),1)    
#     df_temp=pd.concat((pd.DataFrame(arr_xTest2[:,1:]),pd.DataFrame(arr_yTest2),pd.DataFrame(arr_yHat),pd.DataFrame(['N'])),1)    
#     df_temp.columns=arr_columns
#     df_res=pd.DataFrame(np.concatenate((df_res.values,df_temp.values)))
#     df_res.columns=arr_columns
    #try:
    dbl_meanVal=abs((arr_yTest-dbl_yHat)/arr_yTest)[0] 
    dbl_meanCdf=float(ss.norm(0,1).cdf(abs(arr_yTest-dbl_yHat)))
    dbl_predMeanVal = abs((arr_yTest2-arr_yHat)/arr_yTest2)[1]
    #arr_chromeEval.append(ols_res.rsquared_adj) #Y의 training 값과 예측값의 R2 값
    #arr_chromeEval.append(dbl_meanVal) #예측값과 실제 Y값의 차이의 비율
    logging.debug('rsqure:'+str(dbl_rSquare), 'meanVal:'+str(dbl_meanVal))
    if np.isnan(dbl_rSquare): 
      dbl_rSquare=-9999.0
    #except: #ZeroDivisionError as e:
    #  dbl_rSquare = -9999.0
    #  logging.error('0으로 나누기 에러' + str(arr_genesTp))
    #dbl_rsquareMean = fnc_calcComplexVal(dbl_rSquare,dbl_meanVal,lst_score)
    #dbl_rsquareMean = fnc_calcComplexVal2(dbl_rSquare,dbl_meanVal,dbl_calcweight)
    dbl_rsquareMean = fnc_calcComplexVal3(dbl_rSquare,dbl_meanCdf,dbl_calcweight)
    arr_chromeEval.append(dbl_rsquareMean) #Fitted 값과 Predict값의 혼
#     df_res['rsquare'] = dbl_rSquare
#     df_res['meanVal'] = dbl_meanVal
#     df_res['predval'] = dbl_predMeanVal
#     df_res['rsquareMean'] = dbl_rsquareMean
#     df_res['uniqnum'] = int_uniqNum
#     df_res['datefrto'] = dateFrTo
#     df_res['popnum'] = int_popNum
    lst_res.append(dateFrTo)
    lst_res.append(int_uniqNum)
    lst_res.append(int_popNum)
    lst_res.append(arr_genes)
    lst_res.append(dbl_rSquare)
    lst_res.append(dbl_meanCdf)
    lst_res.append(dbl_calcweight)
    lst_res.append(dbl_rsquareMean)
    lst_res.append(arr_yTest2[-1])
    lst_res.append(arr_yHat[-1])
    lst_res.append(abs(arr_yTest2[-1]-arr_yHat[-1]))
    lst_res.append(dbl_predCoef)
    lst_res.append(arr_yTest2)
    lst_res.append(arr_yHat)
    lst_res.append(str(arr_yHatMonth.tolist())[1:-1])
    lst_res.append(str(arr_yHatWeek.tolist())[1:-1])
    
    lst_resFile.append(lst_res)
        
    int_uniqNum += 1
#     if len(df_resFile) ==0: 
#         df_resFile = df_res.copy()
#     else:
#         df_resFile = df_resFile.append(df_res)
        
  return arr_chromeEval

def fnc_calcComplexVal3(dbl_rSquare, dbl_meanCdf,dbl_weight):
    dbl_rvsMeanVal = (1.0 - dbl_meanCdf)*2 #CDF는 0.5가 가장 좋고 커질수록 나빠지기 때문에 조정
    return dbl_rSquare*dbl_weight+dbl_rvsMeanVal*(1.0-dbl_weight)
  
def fnc_calcComplexVal2(dbl_rSquare, dbl_meanVal,dbl_weight):
    dbl_rvsMeanVal = 1.0 - dbl_meanVal
    return dbl_rSquare*dbl_weight+dbl_rvsMeanVal*(1.0-dbl_weight)

def fnc_calcComplexVal(dbl_rSquare,dbl_meanVal,lst_score):
  rsquare = int(round(dbl_rSquare,1) * 10)
  meanval = int(round(dbl_meanVal,1) * 10)
  if rsquare < 0: rsquare=-1
  if meanval > 10: meanval=11  
  rsquareScore = lst_score[10-rsquare]
  meanvalScore = lst_score[meanval]
  
  return rsquareScore+meanvalScore

def fnc_genPop(int_curPop):
  global arr_pool, arr_pop, int_numChromesInPop, int_probGeneLive
  if int_curPop == 0: arr_pop=np.array([]) #첫번째 세대인 경우에만 초기화
  #염색체 개수를 생성
  while len(arr_pop) != int_numChromesInPop:
  #for i in range(int_numChromesInPop):
    chrome = arr_pool[~np.random.randint(int_probGeneLive,size=len(arr_pool)).astype(bool)]
    fnc_addChromeToPop(chrome)

  logging.debug(str(int_curPop)+'세대 염색체 수:'+str(len(arr_pop)))
  
def fnc_evolvePop(df_pop,int_curPop):
  global arr_pool,arr_pop, int_numChromesInPop,int_probGeneLive
  arr_pop=np.array([])
  
  #하위 40% 탈락
  df_pop=df_pop.ix[df_pop.sort(columns=['evaluation'],ascending=False).index[:-int(len(df_pop)*0.4)]]
  logging.debug(str(int_curPop)+'세대 잔여 염색체 수:'+str(len(df_pop)))

  #상위 20%는 존속
  for i in range(int(int_numChromesInPop * 0.2)):
    fnc_addChromeToPop(np.array(df_pop.ix[df_pop.index[i]][0]))  
  logging.debug('상위 10% 존속 개수:'+str(len(arr_pop)))

  #새로운 세대를 교배와 신규생성을 통해 작업
  arr_popIdx=df_pop.index
  while len(arr_popIdx) > 1:
    arr_pickIdx = random.sample(arr_popIdx,2)
    #새로운 염색체를 만듬
    arr_newChrome=np.array([])
    for i in arr_pickIdx:
      #logging.debug('교배를 위해 선택된 염색체:'+str(df_pop.ix[i].tolist()))
      for j in df_pop.ix[i][0]:
        if np.random.random() > 0.5 and j not in arr_newChrome: #1/2확률로 추가
            arr_newChrome = np.append(arr_newChrome,j)
    fnc_addChromeToPop(arr_newChrome)  
    logging.debug(str(int_curPop)+'세대 추가 후보 염색체:'+str(pd.DataFrame(arr_newChrome)[0].str[-3:].tolist()))
    arr_popIdx = arr_popIdx - arr_pickIdx
  if len(arr_popIdx) == 1:
    logging.debug('교배되지 못한 염색체:'+str(df_pop.ix[arr_popIdx[0]][0]))
    fnc_addChromeToPop(np.array(df_pop.ix[arr_popIdx[0]][0]))
  logging.debug(str(int_curPop)+'세대 염색체 수(교배후): '+str(len(arr_pop)))
  
def fnc_execGa():
  global int_uniqNum,df_resFile,lst_resFile,period_q,dfm,dfw


  #간격을 10으로 하자
  int_gap = 21
  fnc_init()
  arr_frPoint = np.arange(0,len(df_stat),1)
  #arr_frPoint=[11]  #11은 위기구간 <- 예측률이 떨어질때 #6은 예측률이 잘맞을때1
  #lst_finRes=[]
  for p in arr_frPoint: #기간에 따른 조건 변경
    if p == 29: break #마지막 값에는 GDP가 없기 때문에 계산 안함
    if (p+int_gap) > len(df_stat): continue
    df_statResize = df_stat[p:p+int_gap]
    period_q = pd.to_datetime(df_statResize['date'][-2:])
    dfm = dfm.ix[period_q.values[1]:][1:]
    dfw = dfw.ix[period_q.values[1]:][1:]
	
    logging.info('Starting Point:' + str(p)+'_'+str(list(df_statResize['gdp'])))
    #lst_resFile = []
    fnc_genPop(0)
    #arr_pops = []  
    for i in range(100): # 세대수
      i+=1
      arr_chromeEval=fnc_evaluateChrome(df_statResize,str(p)+'_'+str(int_gap),i)
      df_pop=pd.DataFrame(arr_pop,columns=['chrome']); df_pop['evaluation']=pd.DataFrame(arr_chromeEval)
      t1=df_pop.dropna().sort(columns=['evaluation'],ascending=False)['evaluation']
      #logging.info(str(p)+'_'+str(int_gap)+'기간,'+str(i)+'세대, 통계(최대,최소,평균,상위10평균):'+str(t1.max())+','+str(t1.min())+','+str(t1.mean())+','+str(t1[:10].mean()))
      #arr_pops.append(df_pop)
      fnc_evolvePop(df_pop,i)
      fnc_genPop(i)
      int_uniqNum +=1  
      #세대별 결과 저장
      #lst_res = []
      #for i in range(len(arr_pops)):
        #lst_res.append(arr_pops[i].sort(columns=['evaluation'],ascending=False)['evaluation'][:10].mean())
      #lst_finRes.append([str(p),str(np.mean(lst_res)),str(max(lst_res)),str(min(lst_res))])
#     df_resFile.to_csv('output/df_res20140305_01_'+str(p)+'_'+str(int_gap)+'.csv',sep='\t')
#     df_resFile=pd.DataFrame()
  df_res=pd.DataFrame(lst_resFile)
  df_res.columns=['dateFrTo','no','pop','XsNum','fitcoef','testgap','weight','fittest','gdp','gdppred','predgap','predcoef','gdptest','gdphat','gdpHatMonth','gdpHatWeek']
  df_res.to_csv('../output/resultExpSVM_01_'+time.strftime("%m%d%H%M")+'.csv',sep='\t',index=False)
  return df_res

      
def fnc_test():
  if len(arr_pool) == 0:fnc_init()
  df_pop=fnc_genPop()
  return df_pop

if __name__ == '__main__':
  os.chdir(os.getcwd())
  fnc_execGa()
