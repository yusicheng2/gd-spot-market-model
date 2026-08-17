import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ================= 页面配置 =================
st.set_page_config(page_title="广东园区光储充现货交易风险量化模型", layout="wide")
st.title("⚡ 广东省园区综合能源现货交易风险量化与对冲模型 (专家版 v3.2)")
st.caption("v3.2 更新：落实增量光伏余电上网“二选一”机制 ｜ 严格校准广东光伏1100利用小时 ｜ 新增全生命周期衰减率与各项刚性运维/前期成本")
st.markdown("---")

# ================= 侧边栏：参数输入 =================
st.sidebar.header("📊 园区资产与交易参数设定")

st.sidebar.subheader("1. 物理资产参数")
pv_cap = st.sidebar.slider("光伏装机 (MW)", 0.0, 20.0, 6.0, 0.5)
pv_hours = st.sidebar.number_input("光伏年等效利用小时 (h)", value=1100, step=50, help="广东地区典型值为1000-1100小时")
ess_cap = st.sidebar.slider("储能装机 (MWh)", 0.0, 50.0, 15.0, 1.0)
ess_power = st.sidebar.slider("储能功率 (MW)", 0.0, 20.0, 5.0, 0.5)
ev_cap = st.sidebar.slider("充电桩装机 (kW)", 0, 10000, 4000, 500)
park_load = st.sidebar.slider("园区日均基础负荷 (MWh)", 10, 100, 45, 5)

st.sidebar.subheader("2. 增量光伏余电上网价格模式 (二选一)")
feed_mode = st.sidebar.radio(
    "光伏余电入市结算方案",
    ["竞价成功：增量光伏项目上网电量的80%享受机制电价", "未参与竞价：全额现货市场价"],
    help="依据广东现行政策，竞价成功者享受机制电价；否则余电全额按现货节点电价结算。"
)
mech_price = st.sidebar.number_input("广东机制电价 (元/kWh)", value=0.453, step=0.005)
spot_mean = st.sidebar.slider("现货日前市场均价期望 (元/kWh)", 0.15, 0.55, 0.25, 0.01)
spot_sigma = st.sidebar.slider("现货价格波动率 (Sigma)", 0.05, 0.30, 0.15, 0.01)

st.sidebar.subheader("3. 偏差考核与风险参数 (双细则)")
deviation_sigma = st.sidebar.slider("光伏预测误差标准差 (%)", 2.0, 20.0, 8.0, 1.0) / 100.0
penalty_multiplier = st.sidebar.slider("偏差惩罚倍数 (实时电价)", 1.0, 3.0, 1.5, 0.1)
deviation_threshold = st.sidebar.slider("免考核死区 (%)", 0.0, 10.0, 5.0, 0.5) / 100.0

st.sidebar.subheader("4. 年化校准与压力情景")
annual_factor = st.sidebar.slider("年化折算系数 (雨季/台风季折减)", 0.60, 1.00, 0.80, 0.05)
typhoon_pv_drop = st.sidebar.slider("台风周光伏出力骤降 (%)", 0, 90, 60, 5) / 100.0
typhoon_price_drop = st.sidebar.slider("台风周现货电价骤降 (%)", 0, 90, 70, 5) / 100.0

st.sidebar.subheader("5. 园区收益分成刚性成本 (二选一)")
share_mode = st.sidebar.radio(
    "收益分成计算模式",
    ["模式一：按年总用电量分成", "模式二：按定额折扣优惠"]
)

if share_mode == "模式一：按年总用电量分成":
    share_vol = st.sidebar.number_input("年用电量 (万kWh, 封顶4000)", min_value=0, max_value=4000, value=2500, step=100)
    share_price = st.sidebar.number_input("度电单价 (元/kWh, 封顶0.10)", min_value=0.00, max_value=0.10, value=0.06, step=0.01)
    annual_share_cost = share_vol * share_price  # 单位：万元
else:
    share_fixed = st.sidebar.number_input("年折扣总金额 (万元, 封顶500)", min_value=0, max_value=500, value=150, step=10)
    annual_share_cost = share_fixed  # 单位：万元

# 【新增】物理资产衰减率与刚性前期/运维成本
st.sidebar.subheader("6. 衰减因子与刚性运营成本")
pv_deg = st.sidebar.number_input("光伏组件年均衰减率 (%)", value=0.5, step=0.1)
ess_deg = st.sidebar.number_input("储能电池年衰减率 (%)", value=2.0, step=0.1)
dev_fee = st.sidebar.number_input("园区路条/前期开发费 (万元)", value=200, step=10)
cont_fee = st.sidebar.number_input("不可预见费用 (万元)", value=50, step=10)
land_rent = st.sidebar.number_input("场地租金 (万元/年)", value=10, step=1)
pv_om = st.sidebar.number_input("光伏运维费 (万元/MW/年)", value=5, step=1)
ess_ev_om = st.sidebar.number_input("储能与充电运维费 (万元/年)", value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.info("💡 专家提示：模型已内置完整的20年生命周期现金流推演，衰减因子与刚性支出将直接影响项目的真实动态回本期。")

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
    daily_base_sum = base_pv_curve[:24].sum()
    
    if pv_cap > 0 and daily_base_sum > 0:
        norm_factor = (pv_hours / 365.0) / daily_base_sum
        pv_generation = pv_cap * 1000 * base_pv_curve * norm_factor
        prediction_error = np.random.normal(0, deviation_sigma, hours)
        pv_actual = pv_generation * np.maximum(0, (1 + prediction_error))
        pv_forecast = pv_generation
        
        # 3. 偏差考核罚款
        deviation = np.abs(pv_actual - pv_forecast)
        threshold_kwh = pv_forecast * deviation_threshold
        penalized_deviation = np.maximum(0, deviation - threshold_kwh)
        penalty_cost = penalized_deviation * spot_prices * penalty_multiplier

        # 4. 收益结算
        if "机制电价" in feed_mode:
            mech_revenue = pv_actual * 0.8 * mech_price
            spot_revenue = pv_actual * 0.2 * spot_prices
        else:
            mech_revenue = np.zeros(hours)
            spot_revenue = pv_actual * spot_prices
    else:
        pv_forecast = np.zeros(hours)
        pv_actual = np.zeros(hours)
        mech_revenue = np.zeros(hours)
        spot_revenue = np.zeros(hours)
        penalty_cost = np.zeros(hours)

    # 5. 储能套利
    ess_revenue = np.zeros(hours)
    if ess_cap > 0 and ess_power > 0:
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
        'Mech_Rev': mech_revenue, 'Spot_Rev': spot_revenue,
        'ESS_Rev': ess_revenue, 'Penalty': penalty_cost
    })

df = simulate_market_and_risk()

# ================= 财务与风险指标 =================
total_rev = df['Mech_Rev'].sum() + df['Spot_Rev'].sum() + df['ESS_Rev'].sum()
total_penalty = df['Penalty'].sum()

# 各项刚性成本折算 (单位：元/万元)
fixed_opex_annual = (pv_cap * pv_om) + ess_ev_om + land_rent  # 万元
monthly_share_cost_rmb = (annual_share_cost * 10000.0) / 12.0
monthly_fixed_opex_rmb = (fixed_opex_annual * 10000.0) / 12.0
weekly_share_cost_rmb = (annual_share_cost * 10000.0) * (7.0 / 365.0)
weekly_fixed_opex_rmb = (fixed_opex_annual * 10000.0) * (7.0 / 365.0)

# 模拟期(30天)毛收益与净收益 (单位：元)
sim_gross_rev = total_rev - total_penalty
sim_net_rev = sim_gross_rev - monthly_share_cost_rmb - monthly_fixed_opex_rmb

# 总投资包含前期路条与不可预见费用
capex = (pv_cap * 280.0 + ess_cap * 70.0 + ev_cap * 0.07) + dev_fee + cont_fee

# 20年全生命周期动态推演 (引入衰减与恒定刚性成本)
pv_rev_1 = (df['Mech_Rev'].sum() + df['Spot_Rev'].sum()) * (365/30) * annual_factor / 10000.0
ess_rev_1 = df['ESS_Rev'].sum() * (365/30) * annual_factor / 10000.0
penalty_1 = df['Penalty'].sum() * (365/30) * annual_factor / 10000.0

cumulative_cash = 0.0
payback_years = 0.0
total_net_20y = 0.0

for y in range(1, 21):
    p_factor = (1 - pv_deg / 100.0)**(y - 1)
    e_factor = (1 - ess_deg / 100.0)**(y - 1)

    # 当年毛利 (衰减后的光储收益 扣减 衰减后的预估罚款)
    y_rev = (pv_rev_1 * p_factor) + (ess_rev_1 * e_factor) - (penalty_1 * p_factor)
    
    # 扣减每年固定的刚性支出
    y_net = y_rev - annual_share_cost - fixed_opex_annual
    total_net_20y += y_net
    
    if payback_years == 0:
        cumulative_cash += y_net
        if cumulative_cash >= capex and y_net > 0:
            payback_years = (y - 1) + (capex - (cumulative_cash - y_net)) / y_net

avg_net_20y = total_net_20y / 20.0
year1_net_rev_10k = (pv_rev_1 + ess_rev_1 - penalty_1) - annual_share_cost - fixed_opex_annual

if capex == 0:
    payback_display = "无新增资产"
elif payback_years > 0:
    payback_display = f"{payback_years:.1f} 年"
else:
    payback_display = ">20年 (无法回本)"

# 蒙特卡洛 P5 风险价值 (含所有刚性成本扣减)
np.random.seed(7)
mc_results = []
for _ in range(2000):
    price_f = np.random.normal(1.0, 0.2)
    dev_f = np.abs(np.random.normal(1.0, 0.4))
    sim_gross = (total_rev * price_f) - (total_penalty * dev_f * penalty_multiplier)
    sim_net = sim_gross - monthly_share_cost_rmb - monthly_fixed_opex_rmb
    mc_results.append(sim_net / 10000.0)
mc_arr = np.array(mc_results)
p5_value = np.percentile(mc_arr, 5)
var95 = (sim_net_rev / 10000.0) - p5_value

# 台风周极端压力测试
np.random.seed(99)
t_s = np.arange(7 * 24)
base_pv_curve = np.maximum(0, np.sin((t_s % 24 - 6) * np.pi / 12))
daily_base_sum = base_pv_curve[:24].sum()

if pv_cap > 0 and daily_base_sum > 0:
    norm_factor = (pv_hours / 365.0) / daily_base_sum
    pv_forecast_week = pv_cap * 1000 * base_pv_curve * norm_factor
    pv_actual_week = pv_forecast_week * (1 - typhoon_pv_drop)
    crash_price = np.clip(spot_mean * (1 - typhoon_price_drop), 0.0, 1.5)

    if "机制电价" in feed_mode:
        stress_revenue = (pv_actual_week * 0.8 * mech_price).sum() + (pv_actual_week * 0.2 * crash_price).sum()
    else:
        stress_revenue = (pv_actual_week * crash_price).sum()
        
    stress_deviation = np.maximum(0, (pv_forecast_week - pv_actual_week) - deviation_threshold * pv_forecast_week)
    stress_penalty = (stress_deviation * crash_price * penalty_multiplier).sum()
else:
    stress_revenue = 0.0
    stress_penalty = 0.0

# 台风周同样需要扣减周维度的各项刚性运营成本
stress_net = stress_revenue - stress_penalty - weekly_share_cost_rmb - weekly_fixed_opex_rmb
normal_week_net = sim_net_rev / (30.0 / 7.0)
stress_shrink_pct = ((normal_week_net - stress_net) / normal_week_net * 100.0) if normal_week_net > 0 else 0.0

# ================= 前端可视化 =================
col1, col2, col3, col4 = st.columns(4)
col1.metric("模拟期(单月)净收益", f"{sim_net_rev/10000:.2f} 万元", f"20年均净收益: {avg_net_20y:.1f} 万")
col2.metric("动态回本期(含衰减)", payback_display, f"首年净利润: {year1_net_rev_10k:.1f} 万")
col3.metric("偏差考核总罚款", f"{total_penalty/10000:.2f} 万元", "⚠️ 现货偏差风险", delta_color="inverse")
col4.metric("P5风险价值 (VaR95)", f"{var95:.2f} 万元", f"P5净收益 {p5_value:.1f} 万", delta_color="inverse")

st.markdown("### 📈 广东现货价格波动与光伏余电价格对冲效果图")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=df['Spot_Price'], mode='lines', name='现货节点电价(LMP)', line=dict(color='red', width=1), opacity=0.6))
if "机制电价" in feed_mode:
    fig1.add_hline(y=mech_price, line_dash="dash", line_color="green", annotation_text=f"政策机制电价 ({mech_price}元)")
fig1.update_layout(height=400, xaxis_title="时间 (小时)", yaxis_title="电价 (元/kWh)", template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 🌀 台风周极端压力测试 (广东气象特色)")
st.caption("情景假设：日前按正常天气申报，台风周光伏出力与现货电价双骤降，超死区偏差按惩罚倍数考核（所有场景均已剥离折算后的运营刚性成本）。")
scol1, scol2, scol3 = st.columns(3)
scol1.metric("正常周均净收益", f"{normal_week_net/10000:.2f} 万元")
scol2.metric("台风周净收益", f"{stress_net/10000:.2f} 万元", delta_color="inverse")
scol3.metric("台风周收益缩水幅度", f"{stress_shrink_pct:.1f} %", delta_color="inverse")

fig_s = go.Figure(go.Bar(
    x=['正常周净收益', '台风周净收益', '其中:台风周偏差罚款'],
    y=[normal_week_net/10000, stress_net/10000, stress_penalty/10000],
    marker_color=['#16a34a', '#dc2626', '#f59e0b'],
    text=[f"{normal_week_net/10000:.2f}万", f"{stress_net/10000:.2f}万", f"{stress_penalty/10000:.2f}万"], textposition='auto'
))
fig_s.update_layout(height=350, yaxis_title="万元", template="plotly_white")
st.plotly_chart(fig_s, use_container_width=True)

st.markdown("### ⚖️ 收益构成与扣款瀑布图")
rev_components = {
    '光伏机制电价收益': df['Mech_Rev'].sum() if "机制电价" in feed_mode else 0,
    '光伏现货敞口收益': df['Spot_Rev'].sum(),
    '储能套利收益': df['ESS_Rev'].sum(),
    '偏差考核罚款(扣除)': -df['Penalty'].sum(),
    '园区收益分成(扣除)': -monthly_share_cost_rmb,
    '固定运维与租金(扣除)': -monthly_fixed_opex_rmb
}
# 过滤掉为0的项保持图表整洁
rev_components = {k: v for k, v in rev_components.items() if v != 0}

fig2 = go.Figure(go.Waterfall(
    name="收益瀑布", orientation="v",
    x=list(rev_components.keys()), y=list(rev_components.values()),
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))
fig2.update_layout(title="模拟期(单月)现金流净构成 (单位：元)", yaxis_title="金额", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 🌪️ 极端行情压力测试 (蒙特卡洛直方图)")
st.caption("2000 次模拟下的净收益分布（已扣减单月刚性成本），橙色虚线为 P5 风险价值分位点。")
fig3 = go.Figure(go.Histogram(x=mc_results, nbinsx=50, marker_color='#2563eb'))
fig3.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="亏损警戒线")
fig3.add_vline(x=p5_value, line_dash="dot", line_color="orange", annotation_text="P5分位")
fig3.update_layout(title="净收益概率分布 (万元)", xaxis_title="净收益", yaxis_title="频次", template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# ================= 专家策略与法律边界分析报告 =================
st.markdown("---")
st.header("📜 专家策略与法律边界分析报告")

st.subheader("1. 长协机制与现货敞口对冲策略")
if "机制电价" in feed_mode:
    st.success(f"**✅ 稳健型结算 (竞价成功)**：当前模型严格落实了**增量光伏项目上网电量的80%享受机制电价（{mech_price}元）**。在广东市场午间频现低价甚至负电价的大环境下，这80%的机制电价是项目收益的“压舱石”，能够大幅对冲现货下行风险。建议配合配置足额工商业储能平移至早晚尖峰放电。")
else:
    st.warning(f"**⚠️ 激进型敞口 (全额现货)**：当前模型适用**未参与竞价的全现货结算**。由于广东光伏出力高峰（午间）与现货价格低谷高度重合，纯现货上网将面临极大的收益缩水风险。务必通过储能错峰放电，或积极争取指标参与竞价获取机制电价保障。")

st.subheader("2. 偏差考核风险量化与应对")
st.error(f"**风险警告**：模拟期偏差罚款 **{total_penalty/10000:.2f} 万元**；台风周压力测试显示极端气象下净收益缩水 **{stress_shrink_pct:.1f}%**。技术应对：必须引入 AI 超短期功率预测系统（将日前预测误差控制在死区内），并利用储能进行实时滚动平抑。")

st.subheader("3. 法律边界与 EMC / 绿电购销合同合规要点")
st.info("**⚖️ 综合能源项目法务风险管控清单**")
st.markdown("""
1. **刚性成本与分层结算剥离**：本模型已剔除支付给园区的固定收益分成、场地租金及设备的恒定运维费。实务中应在合同中明示该“固定收益分享额”的触发前提，避免在系统极端亏损且承受高额偏差罚款的情形下被强行抽血。
2. **台风/暴雨不可抗力免责条款**：台风等极端天气导致的出力骤降偏差，属于典型的《民法典》不可抗力事件。在并网协议与购售电合同中必须明确"极端气象"（如红色台风预警）的触发阈值，并依据广东电力市场规则及时申请免除“两个细则”的偏差考核。
3. **偏差罚款责任传导**：管理商不得在合同中单方兜底现货偏差罚款。应明确约定因园区业主负荷设备突发故障导致的大幅用电偏差，其对应的辅助服务摊销或现货罚金由业主方承担。
4. **VPP 聚合与独立储能合规**：如利用园区独立储能打包参与广东现货及辅助服务市场（需求响应），须事前取得业主排他性的调度授权，并厘清聚合商与南网调度机构在指令执行延误时的过错赔偿责任边界。
""")

st.markdown("---")
st.caption("⚖️ 免责声明：本模型已同步当前有效的广东省能源政策及电力市场交易规则（含现货结算模式），测算结果供投资论证及风险对冲参考，不构成法定收益承诺。实际结算数据以广东电力交易中心正式出具的结算单及电网调度指令为准。")
