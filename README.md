# 量化交易回测工具 Quant-Backtest Tool
对单个金融资产或投资组合进行均值回归策略的回测
Backtest 

mean reversion strategy for an asset or a portfolio

---

# 使用说明 Instructions 

本项目实现了一个 MCP 服务器[^MCP]，用于回测不可卖空的均值回归策略（持有与当前价格偏离滑动平均数的滑动标准差倍数成比例的多头头寸），可在单个金融资产或投资组合上运行。

This project implements an MCP (Model Context Protocol) server[^MCP] for backtesting long-only mean-reversion strategies, where position sizes are proportional to the number of standard deviations the current price deviates from its moving average. The server can be applied to both single financial assets and asset portfolios.

[^MCP]: [什么是MCP服务器？What is a MCP server?](https://modelcontextprotocol.io/docs/getting-started/intro)

## MCP 工具 MCP tools

### `get_strategy_types`
列出当前 MCP 支持的策略类型（单资产均值回归和投资组合均值回归）。

List the strategy types currently supported by MCP (single-asset mean reversion and portfolio mean reversion).

### `do_backtest`
对单个资产进行均值回归策略回测，使用 Z-score 建仓，假设无交易成本，输出半衰期和回测总盈利。

Perform backtesting of a mean reversion strategy on a single asset using Z-score-based entry signals, assuming zero transaction costs. Output the half-life and total backtesting profit.

### `do_backtest_portfolio`
对多个资产构成的投资组合进行均值回归策略回测，基于 Johansen 协整检验确定对冲比率（不允许卖空），假设无交易成本，输出协整检验结果、半衰期和回测总结。

Perform backtesting of a mean reversion strategy on a portfolio of multiple assets. Use the Johansen cointegration test to determine hedge ratios (short selling not allowed), assuming zero transaction costs. Output the cointegration test results, half-life, and backtest summary.

### `do_Johansen_test`
对多个时间序列进行 Johansen 协整检验，判断协整关系的个数，输出特征值统计量和迹统计量的结论。

Conduct the Johansen cointegration test on multiple time series to determine the number of cointegrating relationships. Output the eigenvalue statistics and trace statistics conclusions.

### `do_adf_test`
对单个时间序列进行 ADF 检验，判断其平稳性，输出 p 值、显著性结论、ADF 统计量及临界值。

Conduct the Augmented Dickey-Fuller (ADF) test on a single time series to assess stationarity. Output the p-value, significance conclusion, ADF test statistic, and critical values.

### `get_history`
获取指定资产在指定时间区间内的历史行情（开盘价、收盘价、最高价、最低价、成交量），以 JSON 格式返回。

Retrieve historical market data (open, close, high, low, volume) for a specified asset over a given time range, and return the data in JSON format.

---

## 环境准备 Setting Up Environment 

- 安装 Python 3.13+

Install Python 3.13+

- 安装 uv

Install uv

- 下载 Cherry Studio

Download Cherry Studio

- 获取 Google Gemini API key

Obtain Google Gemini API key

---

## 下载代码

```bash
git clone https://github.com/He-Ri-Shi-Gui-Nian/Quant-Backtest.git
cd Quant-Backtest
