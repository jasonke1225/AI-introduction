#%%
### make sure the file in ./primevalData folder should be format {'date, open, high, low, close, volume'}
### come from https://coinmarketcap.com/zh-tw/historical/
### the outpur csv file will be saved in ./usefulData folder

import pandas as pd
from stockstats import StockDataFrame as Sdf
import os

Month2num = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06',\
    'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

### need sourse data folder
def arrangeData(foldername):
    df_return = None
    for filename in os.listdir(foldername+"/"):
            print(filename)
            date_, open_, high_, low_, close_, volume_ = list(), list(), list(), list(), list(), list()
            
            with open(foldername+'/'+filename, 'r') as f:
                s = f.readline()
                index = s.replace('\n','').split('\t')[:6]
                index.insert(1,'tic')

                s = f.readline()
                while(s):
                    data = s.replace('\n','').split('\t')

                    dat = data[0].replace(',','').split(' ')
                    date_.append(dat[2]+Month2num[dat[0]]+dat[1])

                    tmp = [d.replace('NT$','').replace(',','') for d in data[1:-1]]
                    open_.append(float(tmp[0]))
                    high_.append(float(tmp[1]))
                    low_.append(float(tmp[2]))
                    close_.append(float(tmp[3]))
                    volume_.append(int(tmp[4]))
                    s = f.readline()

                tic_list = [filename.replace('.txt','')] * len(volume_)
                
                dfmap = {index[0]:date_, index[1]:tic_list, index[2]:open_, index[3]:high_,\
                    index[4]:low_, index[5]:close_, index[6]:volume_}

                df = pd.DataFrame(data=dfmap)
                df = df.sort_values([index[0]],ignore_index=True)
                stock = Sdf.retype(df.copy())

                df['macd'] = stock['macd'].values
                df['rsi'] = stock['rsi_30'].values
                df['cci'] = stock['cci_30'].values
                df['adx'] = stock['dx_30'].values
                
                df_return = pd.concat([df_return, df]) if df_return is not None else df

    df_return = df_return.sort_values(['Date', 'tic'])
    df_return.to_csv('usefulData/result.csv',index=False)

    return df_return

def getArrangedData():
    df = pd.read_csv('usefulData/result.csv')
    date = df.Date.unique()
    print(date[date>=20150808])
    data = df.loc[df.Date==20150808]
    return df
getArrangedData()