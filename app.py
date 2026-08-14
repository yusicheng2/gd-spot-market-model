import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ================= 页面配置 =================
st.set_page_config(page_title="广东园区光储充现货交易风险量化模型", layout="wide")
st.title("⚡ 广东省园区综合能源现货交易风险量化与对冲模型 (专家版 v2)")
st.caption("v2 更新：修复极端风险指标失效 ｜ 新增台风周压力测试 ｜ 新增年化校准系数与合规免责声明")
st.markdown("---")

# ================= 侧边栏：参数输入 =================
st.sidebar.header("📊 园区资产与交易参数设定")

st.sidebar.subheader("1. 物理资产参数")
pv_cap = st.sidebar.slider("光伏装机 (MW)", 1.0, 20.0, 6.0, 0.5)
ess_cap = st.sidebar.slider("储能装机 (MWh)", 5.0, 50.0, 15.0, 1.0)
ess_power = st.sidebar.slider("储能功率 (MW)", 2.0, 20.0, 5.0, 0.5)
ev_cap = st.sidebar.slider("充电桩装机 (kW)", 500, 10000, 4000, 500)
park_load = st.sidebar.slider("园区日均基础负荷 (MWh)", 10, 100, 45, 5)

st.sidebar.subheader("2. 广东现货市场交易参数 (CfD对冲)")
cfd_ratio = st.sidebar.slider("中长期长协锁定比例 (%)", 0, 100, 80, 5) / 100.0
cfd_price = st.sidebar.slider("长协基准价 (元/kWh)", 0.30, 0.60, 0.45, 0.01)
spot_mean = st.sidebar.slider("现货日前市场均价期望 (元/kWh)", 0.20, 0.55, 0.38, 0.01)
spot_sigma = st.sidebar.slider("现货价格波动率 (Sigma)", 0.05, 0.30, 0.15, 0.01)

st.sidebar.subheader("3. 偏差考核与风险参数 (双细则)")
deviation_sigma = st.sidebar.slider("光伏预测误差标准差 (%)", 2.0, 20.0, 8.0, 1.0) / 100.0
penalty_multiplier = st.sidebar.slider("偏差惩罚倍数 (实时电价)", 1.0, 3.0, 1.5, 0.1)
deviation_threshold = st.sidebar.slider("免考核死区 (%)", 0.0, 10.0, 5.0, 0.5) / 100.0

st.sidebar.subheader("4. 年化校准与压力情景 (v2新增)")
annual_factor = st.sidebar.slider("年化折算系数 (雨季/台风季折减)", 0.60, 1.00, 0.80, 0.05)
typhoon_pv_drop = st.sidebar.slider("台风周光伏出力骤降 (%)", 0, 90, 60, 5) / 100.0
typhoon_price_drop = st.sidebar.slider("台风周现货电价骤降 (%)", 0, 90, 70, 5) / 100.0

st.sidebar.markdown("---")
st.sidebar.info("💡 专家提示：广东现货市场采用节点电价(LMP)，新能源需参与日前/实时市场，偏差考核极为严格。")

# ================= 后端核心计算引擎（蒙特卡洛模拟） =================
def simulate_market_and_risk(days=30, steps=24):
    np.random.seed(42)
    hours = days * steps
    t = np.arange(hours)

    # 1. 现货价格序列（日内周期：午间低谷、早晚高峰）
    daily_cycle = 0.15 * np.sin((t % 24 - 6) * np.pi / 12)
    spot_prices = spot_mean + daily_cycle + np.random.normal(0, spot_sigma, hours)
    spot_prices = np.clip(spot_prices, 0.0, 1.5)

    # 2. 光伏出力与预测偏差
    base_pv_curve = np.maximum(0, np.sin((t % 24 - 6) * np.pi / 12))
    pv_generation = pv_cap * 1000 * base_pv_curve * 0.85
    prediction_error = np.random.normal(0, deviation_sigma, hours)
    pv_actual = pv_generation * (1 + prediction_error)
    pv_forecast = pv_generation

    # 3. 偏差考核罚款（超死区部分按实时电价×惩罚倍数）
    deviation = np.abs(pv_actual - pv_forecast)
    threshold_kwh = pv_forecast * deviation_threshold
    penalized_deviation = np.maximum(0, deviation - threshold_kwh)
    penalty_cost = penalized_deviation * spot_prices * penalty_multiplier

    # 4. 收益结算（长协锁定 + 现货敞口）
    cfd_revenue = pv_forecast * cfd_ratio * cfd_price
    spot_revenue = pv_actual * (1 - cfd_ratio) * spot_prices

    # 5. 储能基于价格信号套利
    ess_revenue = np.zeros(hours)
    soc = ess_cap * 0.5
    for h in range(hours):
        price = spot_prices[h]
        if price < (spot_mean - 0.5 * spot_sigma) and soc < ess_cap * 0.9:
            charge = min(ess_power * 1000, (ess_cap * 0.9 - soc) / 0.85)
            soc += charge * 0.85
            ess_revenue[h] -= charge * price
        elif price > (spot_mean + 0.5 * spot_sigma) and soc > ess_cap * 0.1:
            discharge = min(ess_power * 1000, (soc - ess_cap * 0.1))
            soc -= discharge / 0.85
            ess_revenue[h] += discharge * price

    return pd.DataFrame({
        'Hour': t, 'Spot_Price': spot_prices,
        'PV_Forecast': pv_forecast, 'PV_Actual': pv_actual,
        'CFD_Rev': cfd_revenue, 'Spot_Rev': spot_revenue,
        'ESS_Rev': ess_revenue, 'Penalty': penalty_cost
    })

df = simulate_market_and_risk()

# ================= 财务与风险指标 =================
total_rev = df['CFD_Rev'].sum() + df['Spot_Rev'].sum() + df['ESS_Rev'].sum()
total_penalty = df['Penalty'].sum()
net_rev = total_rev - total_penalty

# 【v2修复②】年化收益引入折算系数
annual_net_rev = net_rev * (365 / 30) * annual_factor / 10000
capex = (pv_cap * 280 + ess_cap * 70 + ev_cap * 0.07)
static_payback = capex / annual_net_rev if annual_net_rev > 0 else float('inf')

# 【v2修复①a】蒙特卡洛 P5 风险价值（替代恒为0的旧回撤算法）
np.random.seed(7)
mc_results = []
for _ in range(2000):
    price_f = np.random.normal(1.0, 0.2)
    dev_f = np.abs(np.random.normal(1.0, 0.4))
    sim = (total_rev * price_f) - (total_penalty * dev_f * penalty_multiplier)
    mc_results.append(sim / 10000)
mc_arr = np.array(mc_results)
p5_value = np.percentile(mc_arr, 5)
var95 = (net_rev / 10000) - p5_value

# 【v2修复①b】台风周极端压力测试（日前按正常天气申报，实际出力骤降）
np.random.seed(99)
t_s = np.arange(7 * 24)
stress_pv_curve = np.maximum(0, np.sin((t_s % 24 - 6) * np.pi / 12))
pv_forecast_week = pv_cap * 1000 * stress_pv_curve * 0.85
pv_actual_week = pv_forecast_week * (1 - typhoon_pv_drop)
crash_price = np.clip(spot_mean * (1 - typhoon_price_drop), 0.0, 1.5)

stress_revenue = (pv_actual_week * cfd_ratio * cfd_price).sum() + \
                 (pv_actual_week * (1 - cfd_ratio) * crash_price).sum()
stress_deviation = np.maximum(0, (pv_forecast_week - pv_actual_week) - deviation_threshold * pv_forecast_week)
stress_penalty = (stress_deviation * crash_price * penalty_multiplier).sum()
stress_net = stress_revenue - stress_penalty

normal_week_net = net_rev / (30 / 7)
stress_shrink_pct = (normal_week_net - stress_net) / normal_week_net * 100 if normal_week_net > 0 else 0.0

# ================= 前端可视化 =================
col1, col2, col3, col4 = st.columns(4)
col1.metric("模拟期净收益", f"{net_rev/10000:.2f} 万元", f"年化约 {annual_net_rev:.1f} 万 (系数{annual_factor:.2f})")
col2.metric("静态回本期", f"{static_payback:.1f} 年" if static_payback != float('inf') else "超20年", "含校准与考核")
col3.metric("偏差考核总罚款", f"{total_penalty/10000:.2f} 万元", "⚠️ 风险敞口", delta_color="inverse")
col4.metric("P5风险价值 (VaR95)", f"{var95:.2f} 万元", f"P5净收益 {p5_value:.1f} 万", delta_color="inverse")

st.markdown("### 📈 现货价格波动与长协基准价对冲效果图")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=df['Spot_Price'], mode='lines', name='现货节点电价(LMP)', line=dict(color='red', width=1), opacity=0.6))
fig1.add_hline(y=cfd_price, line_dash="dash", line_color="green", annotation_text=f"长协基准价 ({cfd_price}元)")
fig1.update_layout(height=400, xaxis_title="时间 (小时)", yaxis_title="电价 (元/kWh)", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 🌀 台风周极端压力测试 (v2新增)")
st.caption("情景假设：日前按正常天气申报，台风周光伏出力与现货电价双骤降，超死区偏差按实时电价×惩罚倍数考核。")
scol1, scol2, scol3 = st.columns(3)
scol1.metric("正常周净收益", f"{normal_week_net/10000:.2f} 万元")
scol2.metric("台风周净收益", f"{stress_net/10000:.2f} 万元", delta_color="inverse")
scol3.metric("收益缩水幅度", f"{stress_shrink_pct:.1f} %", delta_color="inverse")
fig_s = go.Figure(go.Bar(
    x=['正常周净收益', '台风周净收益', '其中:台风周偏差罚款'],
    y=[normal_week_net/10000, stress_net/10000, stress_penalty/10000],
    marker_color=['#16a34a', '#dc2626', '#f59e0b']
))
fig_s.update_layout(height=350, yaxis_title="万元", template="plotly_white")
st.plotly_chart(fig_s, use_container_width=True)

st.markdown("### ⚖️ 收益构成与偏差罚款瀑布图")
rev_components = {
    '长协锁定收益': df['CFD_Rev'].sum(),
    '现货敞口收益': df['Spot_Rev'].sum(),
    '储能套利收益': df['ESS_Rev'].sum(),
    '偏差考核罚款(扣除)': -df['Penalty'].sum()
}
fig2 = go.Figure(go.Waterfall(
    name="收益瀑布", orientation="v",
    x=list(rev_components.keys()), y=list(rev_components.values()),
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))
fig2.update_layout(title="模拟期现金流构成 (单位：元)", yaxis_title="金额", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 🌪️ 极端行情压力测试 (蒙特卡洛直方图)")
st.caption("2000 次模拟下的净收益分布，橙色虚线为 P5 风险价值分位点。")
fig3 = go.Figure(go.Histogram(x=mc_results, nbinsx=50, marker_color='#2563eb'))
fig3.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="亏损警戒线")
fig3.add_vline(x=p5_value, line_dash="dot", line_color="orange", annotation_text="P5分位")
fig3.update_layout(title="净收益概率分布 (万元)", xaxis_title="净收益", yaxis_title="频次", template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# ================= 专家策略与法律边界分析报告 =================
st.markdown("---")
st.header("📜 专家策略与法律边界分析报告")

st.subheader("1. 长协与现货敞口对冲策略建议")
if cfd_ratio >= 0.8:
    st.success(f"**✅ 策略评价：稳健型 (防御性对冲)** —— 当前长协锁定比例 **{cfd_ratio*100:.0f}%**，有效屏蔽午间零/负电价风险。建议保留 10%-15% 现货敞口，结合储能于早晚高峰放电套利。")
else:
    st.warning(f"**⚠️ 策略评价：激进型 (暴露于现货波动)** —— 当前长协锁定比例仅 **{cfd_ratio*100:.0f}%**。建议提高至 85% 以上，或引入虚拟电厂(VPP)聚合对冲。")

st.subheader("2. 偏差考核风险量化与应对")
st.error(f"**风险警告**：模拟期偏差罚款 **{total_penalty/10000:.2f} 万元**；台风周压力测试显示净收益缩水 **{stress_shrink_pct:.1f}%**。技术应对：引入 AI 超短期功率预测（误差<3%）+ 储能实时滚动平抑偏差。")

st.subheader("3. 法律边界与 EMC 合同风险传导")
st.info("**⚖️ 法律合规提示 (园区综合能源管理商必看)**")
st.markdown("""
1. **不可抗力免责条款**：台风等极端天气导致的出力骤降偏差，应依据《民法典》及广东市场规则申请考核豁免，EMC 合同中须明确"极端气象"触发阈值。
2. **偏差罚款风险传导**：严禁管理商单方兜底现货偏差罚款，应在合同中约定因业主负荷突变导致的罚款由业主承担或从分成中扣减。
3. **VPP 聚合授权**：打包参与现货/辅助服务市场须取得业主排他性调度授权，厘清聚合商与调度机构的主体责任边界。
""")

st.markdown("---")
st.caption("⚖️ 免责声明：本模型基于蒙特卡洛算法与典型气象/电价参数推演，结果仅供商业决策参考，不构成投资收益保证；实际结算以广东电力交易中心规则及电网调度指令为准。演示参数，实际项目请以可行性研究报告为准。")
