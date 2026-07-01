# -*- coding: utf-8 -*-
"""阶段A装配 自查脚本：把 9 个源文件重新装配一遍，断言与已存 Y_oil.csv/g_oil.csv 完全一致。
用法：把 main 上 9 个源 Excel 放到本目录的 raw/ 子文件夹，然后运行  python verify_assembly.py
需要 pandas、openpyxl。"""
import pandas as pd, numpy as np, os, sys
RAW='raw'
CN=['巴西','加拿大','中国','法国','德国','印度','印度尼西亚','意大利','日本','韩国','墨西哥','俄罗斯','南非','土耳其','英国','美国']
EN=['BR','CA','CN','FR','DE','IN','ID','IT','JP','KR','MX','RU','ZA','TR','GB','US']
FULL=pd.period_range('2005-07','2026-02',freq='M')
def load(f):
    d=pd.read_excel(os.path.join(RAW,f)); d=d.rename(columns={d.columns[0]:'date'})
    d['date']=pd.to_datetime(d['date']).dt.to_period('M'); return d.set_index('date').reindex(FULL)
cty={'CPI':'CPI.xlsx','IP':'工业生产指数.xlsx','RATE':'政策利率_日本缺失赋值为0.xlsx','NEER':'名义有效汇率.xlsx'}
glob=[('oil_prod','全球原油产量_数据处理后.xlsx','全球原油产量'),('IGREA','全球实际经济活动指数.xlsx','全球实际经济活动指数'),
      ('loil_price','对数实际油价.xlsx','对数实际油价'),('GSCPI','全球供应链压力指数GSCPI.xlsx','GSCPI'),
      ('GEPU','全球经济政策不确定指数GEPU.xlsx','GEPU_current')]
Y=pd.DataFrame(index=FULL)
for blk,f in cty.items():
    s=load(f); assert list(s.columns)==CN, f'{f} 国名/序不符'
    for cn,en in zip(CN,EN): Y[f'{blk}_{en}']=s[cn].values
g=pd.DataFrame(index=FULL)
for name,f,col in glob: g[name]=load(f)[col].values
Yf=Y.interpolate('linear',limit_direction='both'); gf=g.interpolate('linear',limit_direction='both')
# 与已存CSV对比
Ys=pd.read_csv('Y_oil.csv',index_col=0); Ys.index=FULL
gs=pd.read_csv('g_oil.csv',index_col=0); gs.index=FULL
ok = np.allclose(Yf.values,Ys.values,atol=1e-9) and np.allclose(gf.values,gs.values,atol=1e-9)
ok = ok and list(Yf.columns)==list(Ys.columns) and list(gf.columns)==list(gs.columns)
ok = ok and Yf.shape==(248,64) and gf.shape==(248,5) and Yf.isna().sum().sum()==0
print('自查结果:', '通过 ✅ 重装配与已存CSV完全一致' if ok else '失败 ❌ 存在不一致')
sys.exit(0 if ok else 1)
