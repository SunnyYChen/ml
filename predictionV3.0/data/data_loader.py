import tushare as ts
import pandas as pd

ts.set_token("eaa82f471985fb6597991be445652385a246054b3e1b042f480994d1")
pro = ts.pro_api()

def get_code_list(date='20150202'):
    # 默认2010年开始回测
    # 通过价格、市值、市盈率和股息率指标的设置，选择个股进行量化回测。
    dd = pro.daily_basic(trade_date=date)
    x1 = dd.close < 100
    # 流通市值低于300亿大于50亿
    x2 = dd.circ_mv > 500000
    x3 = dd.circ_mv < 3000000
    # 市盈率低于80
    x4 = dd.pe_ttm < 80
    # 股息率大于2%
    x5 = dd.dv_ttm > 3
    x = x1 & x2 & x3 & x4 & x5
    stock_list = dd[x].ts_code.values
    return stock_list

# 日行情
def get_daily(tsCode, startTime):
    df = pro.daily(ts_code=tsCode, start_date=startTime)
    df.set_index(["trade_date"], inplace=True)
    df.drop(['ts_code'], 1, inplace=True)
    df.sort_index(inplace=True)
    return df


# 日基本面数据
def get_daily_basic(tsCode, startTime):
    df = pro.daily_basic(ts_code=tsCode, start_date=startTime)
    df.set_index(["trade_date"], inplace=True)
    df.drop(['ts_code'], 1, inplace=True)
    df.sort_index(inplace=True)
    return df


# 日资金流转
def get_monney_flow(tsCode, startTime):
    df = pro.moneyflow(ts_code=tsCode, start_date=startTime)
    df.set_index(["trade_date"], inplace=True)
    df.drop(['ts_code'], 1, inplace=True)
    df.sort_index(inplace=True)
    return df


# 合成的数据
def load_combined_data(tsCode, startTime):
    # 读入日行情数据
    df_daily = get_daily(tsCode, startTime)
    # 读入日基本面数据
    df_daily_basic = get_daily_basic(tsCode, startTime)
    # 读入资金流数据
    df_money_flow = get_monney_flow(tsCode, startTime)

    # 按照日期拼接数据
    df = df_daily
    cols_to_use = df_daily_basic.columns.difference(df.columns)
    df = pd.merge(df_daily, df_daily_basic[cols_to_use], how='left', on='trade_date', indicator=False)
    cols_to_use = df_money_flow.columns.difference(df.columns)
    df = pd.merge(df, df_money_flow[cols_to_use], how='left', on='trade_date')
    df.rename(columns={'vol': 'volume'}, inplace=True)

    return df

def data_loader():
    # ts_codes = ["600519.SH", "000002.SZ", "002594.SZ", "603259.SH", "600436.SH", "603027.SH", "002475.SZ",
    #            "000651.SZ", "600031.SH", "002142.SZ", "600030.SH", "600362.SH", "600547.SH", "601111.SH", "002352.SZ",
    #            "002079.SZ", "600739.SH", "300331.SZ", "300382.SZ", "300628.SZ", "600844.SH", "300761.SZ"]
    # ts_codes = ["000651.SZ"]
    # ts_codes = ['600584.SH', '600745.SH', '603501.SH']
    ts_codes = ["000002.SZ"]
    # ts_codes = get_code_list()
    for ts_code in ts_codes:
        start_time = "20150101"
        # 调用API获取原始数据
        raw_df = load_combined_data(ts_code, start_time)
        raw_df.to_csv('%s.csv' % ts_code)