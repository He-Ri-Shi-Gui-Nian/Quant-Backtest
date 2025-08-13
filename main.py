from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import numpy as np

# Create an MCP server
mcp = FastMCP("MCP-backtest")




@mcp.tool()
def get_strategy_types() -> str:
    """
    This function lists the currently available strategy categories
    """
    message = ''' 
This MCP supports the following strategies: 
1. Mean Reversion for an asset
2. Mean Reversion for a portfolio of assets
    '''
    return message



@mcp.tool()
def do_backtest(
    ticker: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict:
    """
    回测均值回复策略(单个资产)
    Z-score建仓
    假设无交易成本
    """
    result_dict = {}


    dat = yf.Ticker(ticker)
    df = dat.history(
        start=train_start, 
        end=train_end, 
        interval=interval, 
        auto_adjust=auto_adjust
    )

    if df is None or df.empty:
        return '该股票不存在. 考虑其是否已退市. '

    # 计算均值回复的半衰期
    df['Close_shifted'] = df['Close'].shift(1)
    df['Dy'] = df['Close'] - df['Close_shifted']
    X = df[['Close_shifted']]
    X = sm.add_constant(X)
    X = X.dropna(subset=['Close_shifted','const'])
    Y = df[['Dy']]
    Y = Y.dropna(subset=['Dy'])
    model = sm.OLS(Y,X)
    results = model.fit()
    Lambda = results.params.loc['Close_shifted']
    halflife = round(-np.log(2)/Lambda)
    result_dict["均值回复的半衰期"] = f"{halflife}天"

    # 建仓回测
    result_dict["回测结果"] = {}
    lookback = halflife  # 滑动窗口 = 半衰期
    test_df = dat.history(
        start=test_start, 
        end=test_end, 
        interval=interval, 
        auto_adjust=auto_adjust
    )

    holding = 0
    cost = 0
    sell_value = 0
    holding_value = 0 
    for r in range(lookback,test_df.shape[0]):
        moving_average = test_df.iloc[r-lookback:r]['Close'].mean()
        moving_std = test_df.iloc[r-lookback:r]['Close'].std()
        price = test_df.iloc[r-1]['Close']
        open_price = test_df.iloc[r]['Open']
        close_price = test_df.iloc[r]['Close']
        # print(f'open price = {open_price}')
        date = str(test_df.index[r])[:10]
        # print(date)
        deviation = (price - moving_average) / moving_std
        if deviation < 0:
            # 买
            if - deviation > holding:
                buy = - deviation - holding
                result_dict["回测结果"][date] = f"以开盘价{round(open_price,2)}美元, 买 {buy} 单位"
                # print(f'deviation = {deviation}')
                # print(f'buy {buy} units')
                cost += buy * open_price
                holding = - deviation
            # 卖
            if - deviation < holding:
                sell = holding + deviation
                result_dict["回测结果"][date] = f"以开盘价{round(open_price,2)}美元, 卖 {sell} 单位"
                # print(f'deviation = {deviation}')
                # print(f'sell {sell} units')
                sell_value += sell * open_price
                holding = - deviation
        else: # deviation大于0则平仓
            if holding > 0:
                sell = holding
                result_dict["回测结果"][date] = f"以开盘价{round(open_price,2)}美元, 卖 {sell} 单位, 平仓"
                # print(f'deviation = {deviation}')
                # print(f'LIQUIDATE: sell {sell} units')
                sell_value += sell * open_price
                holding = 0 


        # 赚的钱用来建仓
        if sell_value > 0:
            if sell_value > cost:
                sell_value = sell_value - cost
                cost = 0
            else:
                cost -= sell_value
                sell_value = 0
                

        # 用当天收盘价计算投资组合的市场价值
        holding_value = close_price * holding

        if date in result_dict["回测结果"]:
            result_dict["回测结果"][date] = result_dict["回测结果"][date] + f"; 目前持有{holding}单位的头寸, 以当日收盘价{round(close_price,2)}美元计算, 头寸的价值为{holding_value}"
        else:
            result_dict["回测结果"][date] = f"目前持有{holding}单位的头寸, 以当日收盘价{round(close_price,2)}美元计算, 头寸的价值为{holding_value}"

    result_dict["回测结果"]["总结"] = f"总盈利(以当日收盘价平仓)为{holding*test_df.iloc[-1]['Close'] + sell_value - cost}"
        # print(f"{date}, holding amount: {holding}, holding value: {holding_value}, sell value: {sell_value}, cost: {cost}")
        # print('-'*30)
    # print(f'total profit = {holding*test_df.iloc[-1]['Close'] + sell_value - cost}')
    # print(f'total cost = {cost}')

    
    return result_dict



@mcp.tool()
def do_backtest_portfolio(
    tickers: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict:
    """
    回测均值回复策略(投资组合)
    Z-score建仓 
    Johansen test 之特征向量表示的投资组合
    假设无交易成本
    """
    result_dict = {}


    df_list = []
    for ticker in tickers:
        dat = yf.Ticker(ticker)
        df = dat.history(
            start=train_start, 
            end=train_end, 
            interval=interval, 
            auto_adjust=auto_adjust
        )
        if df is None or df.empty:
            return f"该股票{ticker}不存在. 考虑其是否已退市."
        df = df.reset_index()
        # 统一日期为字符串
        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)
        df = df.rename(columns={"Close": f"Close {ticker}", "Open": f"Open {ticker}"})
        df_list.append([df,ticker])


    # start with the first DataFrame
    merged_df = df_list[0][0][['Date',f'Close {df_list[0][1]}',f'Open {df_list[0][1]}']]          
    for el in df_list[1:]:
        df = el[0]
        ticker = el[1]
        merged_df = pd.merge(merged_df, df[['Date',f'Close {ticker}',f'Open {ticker}']], on='Date')


    # 是否协整
    result_dict["协整检验"] = {
        "特征值": do_Johansen_test(tickers=tickers, start=train_start, end=train_end)["结论"]["特征值统计量 的结论"],
        "迹": do_Johansen_test(tickers=tickers, start=train_start, end=train_end)["结论"]["迹统计量 的结论"]
    }


    # 不论是否协整, 直接回测
    johansen_df = pd.DataFrame()
    for ticker in tickers:
        johansen_df[f'Close {ticker}'] = merged_df[f'Close {ticker}']
    result = coint_johansen(johansen_df, det_order=0, k_ar_diff=1)

    hedge_ratio = []
    for vec in result.evec:
        if min(list(vec)) > 0:
            hedge_ratio = list(vec)
            break
    if len(hedge_ratio) == 0:
        return "在不卖空的情况下, 不存在一组合适的对冲比率"

    
    df = pd.DataFrame({
        'Date': merged_df['Date'],
        'Open': 0.0,
        'Close': 0.0
    })
    for i, ticker in enumerate(tickers):
        df['Open'] += hedge_ratio[i] * merged_df[f'Open {ticker}']
        df['Close'] += hedge_ratio[i] * merged_df[f'Close {ticker}']

    # 计算均值回复的半衰期
    df['Close_shifted'] = df['Close'].shift(1)
    df['Dy'] = df['Close'] - df['Close_shifted']
    X = df[['Close_shifted']]
    X = sm.add_constant(X)
    X = X.dropna(subset=['Close_shifted','const'])
    Y = df[['Dy']]
    Y = Y.dropna(subset=['Dy'])
    model = sm.OLS(Y,X)
    results = model.fit()
    Lambda = results.params.loc['Close_shifted']
    halflife = round(-np.log(2)/Lambda)
    result_dict["均值回复的半衰期"] = f"{halflife}天"

    # 建仓回测
    result_dict["回测结果"] = {}
    lookback = halflife  # 滑动窗口 = 半衰期
    test_df = dat.history(
        start=test_start, 
        end=test_end, 
        interval=interval, 
        auto_adjust=auto_adjust
    )

    holding = 0
    cost = 0
    sell_value = 0
    holding_value = 0 
    for r in range(lookback,test_df.shape[0]):
        moving_average = test_df.iloc[r-lookback:r]['Close'].mean()
        moving_std = test_df.iloc[r-lookback:r]['Close'].std()
        price = test_df.iloc[r-1]['Close']
        open_price = test_df.iloc[r]['Open']
        close_price = test_df.iloc[r]['Close']
        date = str(test_df.index[r])[:10]
        deviation = (price - moving_average) / moving_std
        # print(date)
        # print(f'open price = {open_price}')
        if deviation < 0:
            # 买
            if - deviation > holding:
                buy = - deviation - holding
                result_dict["回测结果"][date] = f"以开盘价{round(open_price,2)}美元, 买 {buy} 单位"
                cost += buy * open_price
                holding = - deviation
            # 卖
            if - deviation < holding:
                sell = holding + deviation
                result_dict["回测结果"][date] = f"以开盘价{round(open_price,2)}美元, 卖 {sell} 单位"
                sell_value += sell * open_price
                holding = - deviation
        else: # deviation大于0则平仓
            if holding > 0:
                sell = holding
                result_dict["回测结果"][date] = f"以开盘价{round(open_price,2)}美元, 卖 {sell} 单位, 平仓"
                sell_value += sell * open_price
                holding = 0 


        # 赚的钱用来建仓
        if sell_value > 0:
            if sell_value > cost:
                sell_value = sell_value - cost
                cost = 0
            else:
                cost -= sell_value
                sell_value = 0
                

        # 用当天收盘价计算投资组合的市场价值
        holding_value = close_price * holding

        if date in result_dict["回测结果"]:
            result_dict["回测结果"][date] = result_dict["回测结果"][date] + f"; 目前持有{holding}单位的头寸, 以当日收盘价{round(close_price,2)}美元计算, 头寸的价值为{holding_value}"
        else:
            result_dict["回测结果"][date] = f"目前持有{holding}单位的头寸, 以当日收盘价{round(close_price,2)}美元计算, 头寸的价值为{holding_value}"
        

    result_dict["回测结果"]["总结"] = f"总盈利(以当日收盘价平仓)为{holding*test_df.iloc[-1]['Close'] + sell_value - cost}"


    return result_dict



@mcp.tool()
def crit_vals(stat,cvs):
    cv90 = cvs[0]
    cv95 = cvs[1]
    cv99 = cvs[2]
    if stat <= cv90:
        return "无法拒绝原假设"
    elif cv90 < stat <= cv95:
        return "90%置信水平下拒绝原假设"
    elif cv95 < stat <= cv99:
        return "95%置信水平下拒绝原假设"
    elif cv99 < stat:
        return "99%置信水平下拒绝原假设"



@mcp.tool()
def do_Johansen_test(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict:
    """
    用途: 判断多条时间序列数据是否协整
    Fetch historical OHLCV and return the result of Johansen-test
    进行Johansen test, 判断协整关系的数量, 如果数量为0则不存在协整关系, 如果数量不为0则存在协整关系
    有截距项, 无线性趋势, 无滞后项
    """

    df_list = []
    for ticker in tickers:
        dat = yf.Ticker(ticker)
        df = dat.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
        df_list.append(df)

        if df is None or df.empty:
            return "The ticker {ticker} doesn't exist. Maybe it got delisted."

    cols = [c for c in ["Date","Open","High","Low","Close","Adj Close","Volume"] if c in df_list[0].columns]
    if "Adj Close" in cols:
        close_col = "Adj Close"
    else:
        close_col = "Close"


    df_list = []
    for ticker in tickers:
        dat = yf.Ticker(ticker)
        df = dat.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
        df = df.reset_index()
        # 统一日期为字符串（避免 datetime 直接进 JSON）
        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)
        df = df.rename(columns={close_col: f"{close_col} {ticker}"})
        df_list.append([df,ticker])


    # start with the first DataFrame
    merged_df = df_list[0][0][['Date',f'{close_col} {df_list[0][1]}']]          
    for el in df_list[1:]:
        df = el[0]
        ticker = el[1]
        merged_df = pd.merge(merged_df, df[['Date',f'{close_col} {ticker}']], on='Date')


    df = merged_df.drop(columns=['Date'])


    result = coint_johansen(df, det_order=0, k_ar_diff=1)


    result_dict = {}
    # print("特征值统计量: ")
    # print('-'*30)
    cannot_reject = -1
    for i in range(len(result.max_eig_stat)):
        stat = result.max_eig_stat[i]
        crit_values = result.max_eig_stat_crit_vals[i]
        null_hypo = crit_vals(stat, crit_values)
        # print(f"H0: rank={i}")
        # print(f"stat = {stat}")
        # print(f"crit_values = {crit_values}")
        # print(null_hypo)
        # print('-'*30)
        if null_hypo == "无法拒绝原假设" and cannot_reject == -1:
            cannot_reject = i
    if cannot_reject == -1:
        # print(f"特征值统计量 的结论: 有{len(result.max_eig_stat)}个协整关系")
        result_dict["特征值统计量 的结论"] = f"有{len(result.max_eig_stat)}个协整关系"
    else:
        # print(f"特征值统计量 的结论: 有{cannot_reject}个协整关系")
        result_dict["特征值统计量 的结论"] = f"有{cannot_reject}个协整关系"



    # print('\n\n')
    # print("迹统计量: ")
    # print('-'*30)
    cannot_reject = -1
    for i in range(len(result.trace_stat)):
        stat = result.trace_stat[i]
        crit_values = result.trace_stat_crit_vals[i]
        null_hypo = crit_vals(stat, crit_values)
        # print(f"H0: rank={i}")
        # print(f"stat = {stat}")
        # print(f"crit_values = {crit_values}")
        # print(null_hypo)
        # print('-'*30)
        if null_hypo == "无法拒绝原假设" and cannot_reject == -1:
            cannot_reject = i
    if cannot_reject == -1:
        # print(f"迹统计量 的结论: 有{len(result.max_eig_stat)}个协整关系")
        result_dict["迹统计量 的结论"] = f"有{len(result.max_eig_stat)}个协整关系"
    else:
        # print(f"迹统计量 的结论: 有{cannot_reject}个协整关系")
        result_dict["迹统计量 的结论"] = f"有{cannot_reject}个协整关系"


    return {
        "回归方程": "有截距项, 无趋势项", 
        "滞后数": 1,
        "结论": result_dict
    }

    


@mcp.tool()
def do_adf_test(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict:
    """
    Fetch historical OHLCV and return the result of ADF-test (mu != 0, beta = 0, lag = 1).

    In statistics, an augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. The alternative hypothesis depends on which version of the test is used, but is usually stationarity or trend-stationarity. It is an augmented version of the Dickey–Fuller test for a larger and more complicated set of time series models.
    """
    dat = yf.Ticker(ticker)
    df = dat.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)

    if df is None or df.empty:
        return "The ticker doesn't exist. Maybe it got delisted."

    # 把 DataFrame 的日期索引（DatetimeIndex）转换成普通列，方便后续转成 JSON
    df = df.reset_index()
    # 统一日期为字符串（避免 datetime 直接进 JSON）
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)

    # 有些版本列名是 "Adj Close"，不存在就忽略
    cols = [c for c in ["Date","Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    rows = df[cols]


    if "Adj Close" in cols:
        price_series = pd.Series(df["Adj Close"])
    else:
        price_series = pd.Series(df["Close"])

    result = adfuller(price_series, maxlag=1, regression='c', autolag=None)
    pvalue = result[1]
    comment = ""
    if pvalue < 0.01:
        comment = "在1%显著性水平下显著"
    elif 0.01 <= pvalue and pvalue < 0.05:
        comment = "在5%显著性水平下显著"
    elif 0.05 <= pvalue and pvalue < 0.1:
        comment = "在10%显著性水平下显著"
    else:
        comment = "不显著"


    return {
        "回归方程:": "有常数项, 无趋势项", 
        "滞后数:": result[2],
        "使用的样本数:": result[3],
        "p-value:": pvalue,
        "显著性": comment,
        "ADF 统计量 与 临界值:": {"统计量": result[0], "临界值": result[4]}
    }




@mcp.tool()
def get_history(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> dict:
    """
    Fetch historical OHLCV and return JSON-serializable data.
    """
    dat = yf.Ticker(ticker)
    df = dat.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)

    if df is None or df.empty:
        return "该股票不存在. 考虑其是否已退市. "

    # 把 DataFrame 的日期索引（DatetimeIndex）转换成普通列，方便后续转成 JSON
    df = df.reset_index()
    # 统一日期为字符串（避免 datetime 直接进 JSON）
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)

    # 有些版本列名是 "Adj Close"，不存在就忽略
    cols = [c for c in ["Date","Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    rows = df[cols].to_dict(orient="records")

    return {
        "ticker": ticker,
        "interval": interval,
        "start": start,
        "end": end,
        "rows": rows,
    }



if __name__ == "__main__":
    mcp.run(transport='stdio')
