from stockstats import StockDataFrame as sdf


# 特征指标说明
# open:开盘价  high:最高价    low:最低价 close:收盘价
# pre_close:昨收价 change:涨跌额  pct_chg:涨跌幅(未复权,如果是复权请用通用行情接口)
# vol:成交量(手)    amount:成交额(千元)  turnover_rate:换手率(%)    turnover_rate_f:换手率(自由流通股)
# volume_ratio:量比   pe:市盈率(总市值/净利润,亏损的PE为空) pe_ttm 市盈率(TTM,亏损的PE为空) pb:市净率(总市值/净资产)
# ps:市销率    ps_ttm 市销率(TTM)  dv_ratio 股息率(%) dv_ttm:股息率(TTM)(%)  total_share:总股本(万股)
# float_share:流通股本(万股)  free_share:自由流通股本(万)    total_mv:总市值(万元) circ_mv:流通市值(万元)
# buy_sm_vol:小单买入量(手)   sell_sm_vol:小单卖出量(手)    buy_sm_amount:小单买入金额(万元)    sell_sm_amount:小单卖出金额(万元)
# buy_md_vol:中单买入量(手)   sell_md_vol:中单卖出量(手)    buy_md_amount:中单买入金额(万元)    sell_md_amount:中单卖出金额(万元)
# buy_lg_vol:大单买入量(手)   sell_lg_vol:大单卖出量(手)    buy_lg_amount:大单买入金额(万元)    sell_lg_amount:大单卖出金额(万元)
# buy_elg_vol:特大单买入量(手) sell_elg_vol:特大单卖出量(手)  buy_elg_amount:特大单买入金额(万元)  sell_elg_amount:特大单卖出金额(万元)
# net_mf_vol:净流入量(手)    net_mf_amount:净流入额(万元)
# ma5:5日均线  ma20:20日均线  ma50:50日均线
# 以下新特征基于上述特征计算 这个做法可能在深度学习中意义不大
# ma5_diff:ma5-ma20 ma20_diff:ma20-ma50 ma5_ma20_radio:ma5/ma20 ma20_ma50_radio:ma20/ma50
# lg_vol_radio:buy_lg_vol/sell_lg_vol   elg_vol_radio:buy_elg_vol/sell_elg_vol


# 转换布尔值成0和1
def change_boolean(x):
    if x:
        return 1
    return 0


def label(x):
    if x >= 0:
        return 1
    return 0


# 计算一些新特征
def generate_features(df):
    # 深度拷贝一份 不改变原来的df
    new_df = df.copy(deep=True)
    stockStat = sdf.retype(new_df)
    new_df['volume_delta'] = stockStat['volume_delta']
    # open delta against next 2 day
    # df['open_2_d'] = stockStat['open_2_d']
    # open price change (in percent) between today and the day before yesterday
    # 'r' stands for rate.
    new_df['open_-2_r'] = stockStat['open_-2_r']
    # CR indicator, including 5, 10, 20 days moving average
    new_df['cr'] = stockStat['cr']
    new_df['cr-ma1'] = stockStat['cr-ma1']
    new_df['cr-ma2'] = stockStat['cr-ma2']
    new_df['cr-ma3'] = stockStat['cr-ma3']
    # volume max of three days ago, yesterday and two days later
    new_df['volume_-3,2,-1_max'] = stockStat['volume_-3,2,-1_max']
    # volume min between 3 days ago and tomorrow
    new_df['volume_-3~1_min'] = stockStat['volume_-3~1_min']
    # KDJ, default to 9 days
    new_df['kdjk'] = stockStat['kdjk']
    new_df['kdjd'] = stockStat['kdjd']
    new_df['kdjj'] = stockStat['kdjj']
    # three days KDJK cross up 3 days KDJD
    new_df['kdjk_3_xu_kdjd_3'] = stockStat['kdjk_3_xu_kdjd_3']
    new_df['kdjk_3_xu_kdjd_3'] = new_df['kdjk_3_xu_kdjd_3'].apply(change_boolean)

    # 2 days simple moving average on open price
    new_df['open_2_sma'] = stockStat['open_2_sma']
    # MACD
    new_df['macd'] = stockStat['macd']
    # MACD signal line
    new_df['macds'] = stockStat['macds']
    # MACD histogram
    new_df['macdh'] = stockStat['macdh']
    # bolling, including upper band and lower band
    new_df['boll'] = stockStat['boll']
    new_df['boll_ub'] = stockStat['boll_ub']
    new_df['boll_lb'] = stockStat['boll_lb']
    # close price less than 10.0 in 5 days count
    new_df['close_10.0_le_5_c'] = stockStat['close_10.0_le_5_c']
    # CR MA2 cross up CR MA1 in 20 days count
    new_df['cr-ma2_xu_cr-ma1_20_c'] = stockStat['cr-ma2_xu_cr-ma1_20_c']
    new_df['cr-ma2_xu_cr-ma1_20_c'] = new_df['cr-ma2_xu_cr-ma1_20_c'].apply(change_boolean)
    # 6 days RSI
    new_df['rsi_6'] = stockStat['rsi_6']
    # 12 days RSI
    new_df['rsi_12'] = stockStat['rsi_12']
    # 10 days WR
    new_df['wr_10'] = stockStat['wr_10']
    # 6 days WR
    new_df['wr_6'] = stockStat['wr_6']
    # CCI, default to 14 days
    new_df['cci'] = stockStat['cci']
    # 20 days CCI
    new_df['cci_20'] = stockStat['cci_20']
    # TR (true range)
    new_df['tr'] = stockStat['tr']
    # ATR (Average True Range)
    new_df['atr'] = stockStat['atr']
    # DMA, difference of 10 and 50 moving average
    new_df['dma'] = stockStat['dma']
    # DMI
    # +DI, default to 14 days
    new_df['pdi'] = stockStat['pdi']
    # -DI, default to 14 days
    new_df['mdi'] = stockStat['mdi']
    # DX, default to 14 days of +DI and -DI
    new_df['dx'] = stockStat['dx']
    # ADX, 6 days SMA of DX, same as stockStat['dx_6_ema']
    new_df['adx'] = stockStat['adx']
    # ADXR, 6 days SMA of ADX, same as stockStat['adx_6_ema']
    new_df['adxr'] = stockStat['adxr']
    # TRIX, default to 12 days
    new_df['trix'] = stockStat['trix']
    # MATRIX is the simple moving average of TRIX
    new_df['trix_9_sma'] = stockStat['trix_9_sma']
    # VR, default to 26 days
    new_df['vr'] = stockStat['vr']
    # MAVR is the simple moving average of VR
    new_df['vr_6_sma'] = stockStat['vr_6_sma']
    new_df['close_5_sma'] = stockStat['close_5_sma']
    new_df['close_10_sma'] = stockStat['close_10_sma']
    new_df['close_20_sma'] = stockStat['close_20_sma']
    new_df['ma_5_10_diff']=(new_df['close_5_sma']- new_df['close_10_sma'])/ new_df['close_10_sma']
    new_df['ma_10_20_diff']=(new_df['close_10_sma']- new_df['close_20_sma'])/ new_df['close_20_sma']
    new_df['ma_5_20_diff']=(new_df['close_5_sma']- new_df['close_20_sma'])/ new_df['close_20_sma']

    new_df['sm_vol_diff']= new_df['buy_sm_vol']- new_df['sell_sm_vol']
    new_df['md_vol_diff'] = new_df['buy_md_vol'] - new_df['sell_md_vol']
    new_df['lg_vol_diff']= new_df['buy_lg_vol']- new_df['sell_lg_vol']
    new_df['elg_vol_diff'] = new_df['buy_elg_vol'] - new_df['sell_elg_vol']
    # 生成label标签
    new_df['label'] = new_df['change'].map(label)
    # 丢弃前面20行 有的指标前面是没有的
    new_df = new_df.iloc[20:]
    return new_df