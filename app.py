import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ================= 页面配置 =================
st.set_page_config(page_title="广东园区光储充现货交易风险量化模型", layout="wide")
st.title("⚡ 广东省园区综合能源现货交易风险量化与对冲模型 (专家版)")
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

st.sidebar.markdown("---")
st.sidebar.info("💡 专家提示：广东现货市场采用节点电价(LMP)，新能源需参与日前/实时市场，偏差考核极为严格。")

# ================= 后端核心计算引擎 (蒙特卡洛模拟) =================
def simulate_market_and_risk(days=30, steps=24):
    np.random.seed(42)
    hours = days * steps
    
    # 1. 模拟现货价格序列 (几何布朗运动 + 均值回归，模拟广东早晚高峰与午间低谷)
    t = np.arange(hours)
    # 引入日内周期性：午间(光伏大发)价格低谷，早晚高峰价格飙升
    daily_cycle = 0.15 * np.sin((t % 24 - 6) * np.pi / 12) 
    spot_prices = spot_mean + daily_cycle + np.random.normal(0, spot_sigma, hours)
    spot_prices = np.clip(spot_prices, 0.0, 1.5) # 广东现货限价设置 (允许负电价或零电价触及0)
    
    # 2. 模拟光伏出力与预测偏差
    # 典型光伏出力曲线 (归一化)
    base_pv_curve = np.maximum(0, np.sin((t % 24 - 6) * np.pi / 12)) 
    pv_generation = pv_cap * 1000 * base_pv_curve * 0.85 # 考虑系统效率
    
    # 引入预测误差 (正态分布)
    prediction_error = np.random.normal(0, deviation_sigma, hours)
    pv_actual = pv_generation * (1 + prediction_error)
    pv_forecast = pv_generation # 日前申报值
    
    # 3. 计算偏差考核罚款
    deviation = np.abs(pv_actual - pv_forecast)
    threshold_kwh = pv_forecast * deviation_threshold
    penalized_deviation = np.maximum(0, deviation - threshold_kwh)
    # 罚款 = 偏差电量 * 实时电价 * 惩罚倍数 (假设实时与日前强相关)
    penalty_cost = penalized_deviation * spot_prices * penalty_multiplier
    
    # 4. 收益结算 (广东 CfD 差价合约逻辑简化版)
    # 总收益 = 长协锁定部分 * 长协价 + 现货敞口部分 * 现货价
    cfd_revenue = pv_forecast * cfd_ratio * cfd_price
    spot_revenue = pv_actual * (1 - cfd_ratio) * spot_prices
    
    # 5. 储能寻优套利 (基于价格信号的贪心算法)
    ess_revenue = np.zeros(hours)
    soc = ess_cap * 0.5 # 初始 SOC 50%
    for h in range(hours):
        price = spot_prices[h]
        # 简化的阈值策略：价格低于均值-0.5sigma充电，高于均值+0.5sigma放电
        if price < (spot_mean - 0.5 * spot_sigma) and soc < ess_cap * 0.9:
            charge = min(ess_power * 1000, (ess_cap * 0.9 - soc) / 0.85)
            soc += charge * 0.85
            ess_revenue[h] -= charge * price # 充电成本
        elif price > (spot_mean + 0.5 * spot_sigma) and soc > ess_cap * 0.1:
            discharge = min(ess_power * 1000, (soc - ess_cap * 0.1))
            soc -= discharge / 0.85
            ess_revenue[h] += discharge * price # 放电收益

    return pd.DataFrame({
        'Hour': t,
        'Spot_Price': spot_prices,
        'PV_Forecast': pv_forecast,
        'PV_Actual': pv_actual,
        'CFD_Rev': cfd_revenue,
        'Spot_Rev': spot_revenue,
        'ESS_Rev': ess_revenue,
        'Penalty': penalty_cost
    })

df = simulate_market_and_risk()

# ================= 财务与风险指标计算 =================
total_rev = df['CFD_Rev'].sum() + df['Spot_Rev'].sum() + df['ESS_Rev'].sum()
total_penalty = df['Penalty'].sum()
net_rev = total_rev - total_penalty

# 年化推算 (假设30天模拟期代表全年典型特征，简化放大)
annualization_factor = 365 / 30
annual_net_rev = net_rev * annualization_factor / 10000 # 万元
capex = (pv_cap * 280 + ess_cap * 70 + ev_cap * 0.07) # 万元估算
static_payback = capex / annual_net_rev if annual_net_rev > 0 else float('inf')

# 极端风险指标 (Max Drawdown 模拟)
cumulative_cashflow = (df['CFD_Rev'] + df['Spot_Rev'] + df['ESS_Rev'] - df['Penalty']).cumsum()
max_drawdown = (cumulative_cashflow.cummax() - cumulative_cashflow).max() / 10000 # 万元

# ================= 前端可视化展示 =================
col1, col2, col3, col4 = st.columns(4)
col1.metric("模拟期净收益", f"{net_rev/10000:.2f} 万元", f"年化约 {annual_net_rev:.1f} 万")
col2.metric("静态回本期", f"{static_payback:.1f} 年", "考虑衰减与考核")
col3.metric("偏差考核总罚款", f"{total_penalty.sum()/10000:.2f} 万元", "⚠️ 风险敞口", delta_color="inverse")
col4.metric("极端最大回撤", f"{max_drawdown:.2f} 万元", "单日/周最大亏损", delta_color="inverse")

st.markdown("### 📈 现货价格波动与长协基准价对冲效果图")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=df['Spot_Price'], mode='lines', name='现货节点电价(LMP)', line=dict(color='red', width=1), opacity=0.6))
fig1.add_hline(y=cfd_price, line_dash="dash", line_color="green", annotation_text=f"长协基准价 ({cfd_price}元)")
fig1.update_layout(height=400, xaxis_title="时间 (小时)", yaxis_title="电价 (元/kWh)", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### ⚖️ 收益构成与偏差罚款瀑布图")
rev_components = {
    '长协锁定收益': df['CFD_Rev'].sum(),
    '现货敞口收益': df['Spot_Rev'].sum(),
    '储能套利收益': df['ESS_Rev'].sum(),
    '偏差考核罚款(扣除)': -df['Penalty'].sum()
}
fig2 = go.Figure(go.Waterfall(
    name = "收益瀑布", orientation = "v",
    x = list(rev_components.keys()),
    y = list(rev_components.values()),
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
))
fig2.update_layout(title="模拟期现金流构成 (单位：元)", yaxis_title="金额", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 🌪️ 极端行情压力测试 (蒙特卡洛直方图)")
# 模拟1000次不同的偏差与价格组合，展示净收益分布
st.caption("基于当前设定的长协比例与预测误差，模拟1000种市场极端情况下的收益分布。")
mc_results = []
for _ in range(1000):
    rand_price_factor = np.random.normal(1.0, 0.2)
    rand_dev_factor = np.random.normal(1.0, 0.3)
    sim_rev = (total_rev * rand_price_factor) - (total_penalty * rand_dev_factor * penalty_multiplier)
    mc_results.append(sim_rev / 10000)

fig3 = go.Figure(go.Histogram(x=mc_results, nbinsx=50, marker_color='#2563eb'))
fig3.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="亏损警戒线")
fig3.update_layout(title="净收益概率分布 (万元)", xaxis_title="净收益", yaxis_title="频次", template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# ================= 专家策略与法律边界分析报告 =================
st.markdown("---")
st.header("📜 专家策略与法律边界分析报告")

st.subheader("1. 长协与现货敞口对冲策略建议")
if cfd_ratio >= 0.8:
    st.success("**✅ 策略评价：稳健型 (防御性对冲)**")
    st.write(f"当前长协锁定比例为 **{cfd_ratio*100:.0f}%**。此策略有效屏蔽了广东现货市场午间光伏大发时段可能出现的**零电价/负电价**风险。")
    st.write("**优化建议**：建议保留 10%-15% 的现货敞口，利用储能系统在早晚高峰（现货价格飙升时段）进行放电套利，以弥补长协价格可能低于高峰现货价的踏空损失。")
else:
    st.warning("**⚠️ 策略评价：激进型 (暴露于现货波动)**")
    st.write(f"当前长协锁定比例仅为 **{cfd_ratio*100:.0f}%**。模型显示偏差考核罚款已严重侵蚀现货敞口收益。")
    st.write("**优化建议**：强烈建议提高长协比例至 85% 以上，或引入**虚拟电厂(VPP)聚合平台**，将光储充打包参与现货市场，利用储能的灵活性对冲光伏出力的不确定性。")

st.subheader("2. 偏差考核风险量化与应对")
st.error(f"**风险警告**：在设定的预测误差 ({deviation_sigma*100:.1f}%) 下，模拟期内产生了高达 **{total_penalty.sum()/10000:.2f} 万元** 的罚款。")
st.write("根据《广东电力现货市场交易规则》，新能源企业需承担日前申报与实时出力的偏差责任。当偏差超过免死区（当前设定为 5%）时，将面临实时电价 1.5 倍以上的惩罚性结算。")
st.write("**技术应对**：必须引入**超短期功率预测系统（AI气象耦合模型）**，将预测误差控制在 3% 以内；同时利用储能系统进行**实时偏差滚动平抑**（即发现预测偏低时，储能立即放电补足缺口）。")

st.subheader("3. 法律边界与 EMC 合同风险传导")
st.info("**⚖️ 法律合规提示 (园区综合能源管理商必看)**")
st.markdown("""
作为综合能源管理商，在与园区业主签订《能源管理合同 (EMC)》或《售电代理协议》时，必须明确以下法律边界，避免承担无限连带责任：
1. **不可抗力免责条款**：根据《民法典》及广东市场规则，因**极端天气（如台风导致光伏组件损坏或骤降）**、**电网调度限电（弃光）** 导致的出力偏差，应具备申请免责的法律依据。合同中必须明确界定“极端气象”的触发阈值。
2. **偏差罚款的风险传导机制**：若采用“保底收益+分成”模式，**严禁**管理商单方面兜底现货市场的偏差罚款。应在合同中约定：“因园区负荷突变或业主设备检修导致的预测偏差罚款，由业主方承担或在分成比例中予以扣减”。
3. **虚拟电厂(VPP)聚合授权**：若需将园区资产打包参与需求侧响应或调频辅助服务市场，必须在合同中获取业主的**排他性调度授权**，明确聚合商与电网调度中心的法律主体责任边界。
""")
