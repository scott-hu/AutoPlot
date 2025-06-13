import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import dash
import os
import datetime
from dash import Dash, dcc, html, Input, Output, callback_context
import webbrowser


pio.renderers.default = 'browser'

def calc_longterm_perf(value, benchmark):
    bench_value = benchmark.loc[value.index]
    bench_ret = bench_value.pct_change(1).dropna()
    strategy_ret = value.pct_change(1).dropna()
    exc_value = (strategy_ret - bench_ret + 1).cumprod()
    delta_year = len(value) / 243
    
    annual_ret = (value.iloc[-1] / value.iloc[0]) ** (1 / delta_year) - 1
    annual_vol = strategy_ret.std(ddof=0) * np.sqrt(243)
    sharpe_ratio = annual_ret / annual_vol
    max_drawdown = - (value / value.cummax() - 1).min()
    calmar_ratio = annual_ret / max_drawdown if max_drawdown > 0 else 0  # 卡玛比率（若无回撤，为0）
    annual_bench_ret = (bench_value.iloc[-1] / bench_value.iloc[0]) ** (1 / delta_year) - 1
    annual_exc_ret = exc_value.iloc[-1] ** (1 / delta_year) - 1
    max_exc_drawdown = - (exc_value / exc_value.cummax() - 1).min()
    information_ratio = annual_exc_ret / ((strategy_ret - bench_ret).std(ddof=0) * np.sqrt(243))

    total_performance = pd.Series({
        '年化收益': annual_ret,
        '年化波动': annual_vol,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '卡玛比率': calmar_ratio,
        '年化超额收益': annual_exc_ret,
        '最大超额回撤': max_exc_drawdown,
        '信息比率': information_ratio,
    })
    
    return total_performance

def calc_short_period(price_series, benchmark_series):
    price_series = price_series.sort_index().dropna()
    benchmark_series = benchmark_series.sort_index()
    bench_series = pd.merge(benchmark_series, price_series, how='right', left_index=True, right_index=True).ffill()[benchmark_series.name]
    # bench_value = benchmark_series.loc[price_series.index]
    bench_ret = bench_series.pct_change(1, fill_method=None).fillna(0)
    strategy_ret = price_series.pct_change(1, fill_method=None).fillna(0)
    exc_value = (strategy_ret - bench_ret + 1).cumprod()
    
    pctchg = price_series.iloc[-1] / price_series.iloc[0] - 1
    max_drawdown = - (price_series / price_series.cummax() - 1).min()

    bench_pctchg = bench_series.iloc[-1] / bench_series.iloc[0] - 1
    exc_pctchg = exc_value.iloc[-1] / exc_value.iloc[0] - 1
    max_exc_drawdown = - (exc_value / exc_value.cummax() - 1).min()
    total_performance = pd.Series({
        '涨跌幅': pctchg,
        '最大回撤': max_drawdown,
        '基准涨跌幅': bench_pctchg,
        '超额涨跌幅': exc_pctchg,
        '最大超额回撤': max_exc_drawdown,
    })
    return total_performance

def calc_return_drawdown_over_periods(price_series, benchmark_series):

    price_series = price_series.dropna().sort_index()
    if len(price_series) < 2:
        return pd.DataFrame(index=["成立以来", "最近一月", "最近三月", "最近半年", "今年以来", "最近一年", "最近两年"], columns=["涨跌幅", "最大回撤", "基准涨跌幅", "超额涨跌幅", "最大超额回撤"])
    
    today = price_series.index[-1]
    periods = {
        "成立以来": price_series.index[0],
        "最近一月": today - pd.DateOffset(months=1),
        "最近三月": today - pd.DateOffset(months=3),
        "最近半年": today - pd.DateOffset(months=6),
        "今年以来": pd.Timestamp(year=today.year, month=1, day=1),
        "最近一年": today - pd.DateOffset(years=1),
        "最近两年": today - pd.DateOffset(years=2),
    }
    result = []
    for label, start_date in periods.items():
        # 无足够数据
        if (start_date < price_series.index[0]) or (start_date < benchmark_series.index[0]):
            result.append(pd.Series(name=label))
            continue
        sub_price_series = price_series.loc[start_date:]
        sub_bench_series = benchmark_series.loc[start_date:]
        # 数据量不足
        if (len(sub_price_series) < 2) or (len(sub_bench_series) < 2):
            result.append(pd.Series(name=label))
            continue
        perf = calc_short_period(sub_price_series, sub_bench_series)
        result.append(pd.Series(perf, name=label))
    df = pd.DataFrame(result).map(lambda x: f"{x:.2%}" if not np.isnan(x) else "N/A")
    return df


def get_all_series(data_path):
    file_list = os.listdir(data_path)  # 主文件夹路径
    fund_series = {}
    for fund_name in file_list:
        file_path = os.path.join(data_path, fund_name, "净值数据.xlsx")
        if os.path.exists(file_path):
            cur_series = pd.read_excel(file_path, index_col=0, parse_dates=True).iloc[:, 0]
            cur_series.name = fund_name
            fund_series[fund_name] = cur_series
            
    bench_df = pd.read_excel(os.path.join(data_path, '基准净值数据.xlsx'), index_col=0, parse_dates=True)
    bench_series = {}
    for name in bench_df.columns:
        value = bench_df[name].replace(0, np.NaN).dropna()
        bench_series[name] = value.loc['2010-01-01': ]

    return fund_series, bench_series
    

def make_figure(price_series_list, benchmark_series, axis_series):
    """画图

    Args:
        price_series (List[pd.Series]): 
        benchmark_series (pd.Series): 
        axis_series (pd.Series): 

    Returns:
        fig: 
    """
    
    all_series = price_series_list + [benchmark_series]
    all_series = pd.concat(all_series, axis=1, join="outer").ffill()
    benchmark_series = all_series[benchmark_series.name]
    series_num = len(price_series_list)
    
    bench = benchmark_series / benchmark_series.iloc[0] - 1

    fig = make_subplots(
        rows=2+series_num, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3] + [0.3] * series_num,
        vertical_spacing=0.1,
        specs=[[{"type": "xy"}], [{"type": "xy"}]] + [[{"type": "table"}]] * series_num
    )

    # 主图
    for i in range(series_num):
        series_name = price_series_list[i].name
        cur_price_series = all_series[series_name].replace(0, np.NaN).dropna()
        ret = cur_price_series / cur_price_series.iloc[0] - 1
        fig.add_trace(go.Scatter(x=ret.index, y=ret.values, name=series_name+'_涨跌幅', mode="lines", hovertemplate=" %{y:.2%}"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name='基准涨跌幅', mode="lines", hovertemplate=" %{y:.2%}"), row=1, col=1)
    # 回撤
    for i in range(series_num):
        series_name = price_series_list[i].name
        cur_price_series = all_series[series_name].replace(0, np.NaN).dropna()
        drawdown = cur_price_series / cur_price_series.cummax() - 1
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name=series_name+'_回撤', mode="lines", fill="tozeroy", hovertemplate=" %{y:.2%}"), row=2, col=1)
    # 表格
    for i in range(series_num):
        series_name = price_series_list[i].name
        perf_df = calc_return_drawdown_over_periods(all_series[series_name], benchmark_series)
        table_values_1 = [[metric] + perf_df.loc[metric, :].tolist() for metric in perf_df.index]
        table_values = list(map(list, zip(*table_values_1)))
        
        fig.add_trace(go.Table(
            header=dict(values=[series_name+"指标"] + list(perf_df.columns), fill_color="#E8EAED", align="center"),
            cells=dict(values=table_values, fill_color="white", align="center")), row=3+i, col=1)


    fig.update_layout(
        height=500+series_num*350, width=1050,
        margin=dict(t=60, b=30, l=40, r=20),
        title="收盘价、回撤与区间指标表",
        hovermode="x unified", plot_bgcolor="white"
    )
    fig.update_xaxes(row=1, col=1, showgrid=True, gridcolor='rgba(200,200,200,0.3)', gridwidth=1, layer='below traces', # 放到 trace 后面
                     range=[all_series.index.min(), all_series.index.max()],
                     rangeslider=dict(visible=True, thickness=0.05, range=[axis_series.index.min(), axis_series.index.max()]))
    fig.update_xaxes(row=2, col=1, tickformat="%Y-%m-%d", tickangle=-45, tickfont=dict(size=8),
                     range=[all_series.index.min(), all_series.index.max()])
    # 格式化 y 轴
    fig.update_yaxes(row=1, col=1, tickformat=".0%", showgrid=True, gridcolor='rgba(200,200,200,0.3)', gridwidth=1, layer='below traces')
    fig.update_yaxes(row=2, col=1, tickformat=".0%", rangemode="tozero")
    
    return fig


class MyDashApp:
    app = Dash(__name__)

    def __init__(self, data_path):        
        self.fund_dict, self.bench_dict = get_all_series(data_path)
        self.fund_name_list = list(self.fund_dict.keys())
        self.benchmark_name_list = list(self.bench_dict.keys())
        self.global_start_date = datetime.date(2010, 1, 1)
        
    def config_app(self):
        # 初始化 Dash 应用
        self.app.layout = html.Div([
            html.Div([
                dcc.Graph(
                    id="dd-graph",
                    # 默认只画第一支
                    figure=make_figure(
                        # 1. price_series_list 2. benchmark_series 3. full_series
                        [self.fund_dict[self.fund_name_list[0]]],
                        self.bench_dict[self.benchmark_name_list[0]],
                        self.bench_dict[self.benchmark_name_list[0]]
                    )
                )
            ], style={'display': 'inline-block', 'width': '75%'}),

            html.Div([
                html.Label("请选择标的："),
                dcc.Dropdown(
                    id='fund-dropdown-list',
                    options=[{'label': s, 'value': s} for s in self.fund_name_list],
                    value=[self.fund_name_list[0]],   # 默认选第一支
                    multi=True
                ),
                html.Br(),
                html.Label("请选择基准："),
                dcc.Dropdown(
                    id='benchmark-dropdown',
                    options=[{'label': s, 'value': s} for s in self.benchmark_name_list],
                    value=self.benchmark_name_list[0],   # 默认选第一个
                    multi=False
                ),
                html.Br(),
                html.Label("请选择时间范围："),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=datetime.date(2010,1,1),  # min(df.index.min().date() for df in self.bench_dict.values()),
                    max_date_allowed=max(df.index.max().date() for df in self.bench_dict.values()),
                    start_date=min(df.index.min().date() for df in self.fund_dict.values()),
                    end_date=max(df.index.max().date() for df in self.fund_dict.values()),
                    display_format='YYYY-MM-DD'
                )
            ], style={
                'display': 'inline-block',
                'verticalAlign': 'top',
                'marginLeft': '2%',
                'width': '23%',
                'marginTop': '170px'
            }),
        ])

    # 回调
    @app.callback(
        Output("dd-graph", "figure"),
        Input("fund-dropdown-list", "value"),
        Input("benchmark-dropdown", "value"),   # 新增
        Input("dd-graph", "relayoutData"),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def update_dd(selected_fund_list, selected_benchmark, relayout, start_date, end_date):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        prop = ctx.triggered[0]['prop_id'].split('.')[0]
        # 确定时间区间
        if prop == 'date-picker-range':
            t0, t1 = pd.to_datetime(start_date), pd.to_datetime(end_date)
        else:
            if not relayout:
                return dash.no_update
            xaxis_range = relayout.get('xaxis.range')
            if xaxis_range and isinstance(xaxis_range, (list, tuple)) and len(xaxis_range) == 2:
                start, end = xaxis_range
            else:
                start = relayout.get('xaxis.range[0]')
                end = relayout.get('xaxis.range[1]')
            if start is None or end is None:
                rng = relayout.get("xaxis.range")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    start, end = rng
            t0, t1 = pd.to_datetime(start), pd.to_datetime(end)

        # 切片
        fund_series_list = [my_dash.fund_dict[selected_fund].loc[t0:t1] for selected_fund in selected_fund_list]
        # benchmark series选用，close基准
        benchmark_series = my_dash.bench_dict[selected_benchmark].loc[t0:t1]  
        if all([len(fund_series) < 2 for fund_series in fund_series_list]):
            return dash.no_update
        return make_figure(fund_series_list, benchmark_series, my_dash.bench_dict[my_dash.benchmark_name_list[0]])

    @app.callback(
        Output("date-picker-range", "start_date"),
        Output("date-picker-range", "end_date"),
        Input("benchmark-dropdown", "value"),
        Input("dd-graph", "relayoutData"),
    )
    def update_date_picker_range(selected_benchmark, relayout):
        if not relayout:
            bench_series = my_dash.bench_dict[selected_benchmark]
            start, end = bench_series.index[0], bench_series.index[-1]
            return start, end
        xaxis_range = relayout.get('xaxis.range')
        if xaxis_range and isinstance(xaxis_range, (list, tuple)) and len(xaxis_range) == 2:
            start, end = xaxis_range
        else:
            start = relayout.get('xaxis.range[0]')
            end = relayout.get('xaxis.range[1]')
        if start is None or end is None:
            rng = relayout.get("xaxis.range")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                start, end = rng
        return start, end

if __name__ == "__main__":
    
    data_path = os.path.join(os.path.join(os.path.realpath(__file__), ".."), 'datas')
    my_dash = MyDashApp(data_path)
    my_dash.config_app()
    webbrowser.open("http://127.0.0.1:8051")
    my_dash.app.run(debug=True, port=8051)
