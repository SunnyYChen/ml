import pandas as pd


# 获取有用的特征名称
def get_feature_names():
    # 'open', 'high','low', 'change', 'close','amount'
    # 'label', 'turnover_rate', 'turnover_rate_f', 'pb', 'pe', 'ps', 'net_mf_vol',
    # 'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol', 'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol',
    # 'sell_elg_vol', 'dv_ratio', 'volume', 'volume_ratio', 'pct_chg', 'volume_delta', 'open_-2_r', 'cr',
    # 'cr-ma1', 'cr-ma2', 'cr-ma3', 'volume_-3,2,-1_max', 'volume_-3~1_min', 'kdjk', 'kdjd', 'kdjj',
    # 'kdjk_3_xu_kdjd_3', 'open_2_sma', 'macd', 'macds', 'macdh', 'boll', 'boll_ub', 'boll_lb',
    # 'close_10.0_le_5_c', 'cr-ma2_xu_cr-ma1_20_c', 'rsi_6', 'rsi_12', 'wr_10', 'wr_6', 'cci', 'cci_20', 'tr',
    # 'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'adxr', 'trix', 'trix_9_sma', 'vr', 'vr_6_sma'
    # 'close_5_sma','close_10_sma','close_20_sma'
    # 推荐以下指标
    # 'label', 'close_5_sma', 'close_10_sma', 'close_20_sma', 'macd', 'macds', 'macdh', 'macd', 'macds',
    # 'macdh', 'boll', 'boll_ub', 'boll_lb', 'kdjk', 'kdjd', 'kdjj', 'kdjk_3_xu_kdjd_3', 'rsi_6',
    # 'rsi_12', 'cci', 'cci_20', 'cr-ma1', 'cr-ma2', 'cr-ma3'
    #ma_5_10_diff 55.38
    #ma_5_20_diff 53.85
    #ma_10_20_diff 56.92
    #sm_vol_diff 55.38
    #md_vol_diff 50.77
    #lg_vol_diff 47.69
    #elg_vol_diff 58.46
    #net_mf_vol 50.77
    #amount 56.92
    feature_names = ['label', 'amount','ma_5_10_diff','ma_10_20_diff','sm_vol_diff','elg_vol_diff']
    return feature_names
