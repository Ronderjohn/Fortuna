//@version=6
strategy(
     title = "Institutional Liquidity Sweep + VWAP Reclaim Strategy",
     shorttitle = "LS-VWAP Institutional",
     overlay = true,
     initial_capital = 500000,
     pyramiding = 0,
     process_orders_on_close = true,
     commission_type = strategy.commission.percent,
     commission_value = 0.03
)

// =====================================================
// INPUTS
// =====================================================

// Trend Settings
emaLength          = input.int(20, "EMA Length")
atrLength          = input.int(14, "ATR Length")
atrSLMultiplier    = input.float(1.0, "ATR Stop Multiplier", step = 0.1)

// Volume Filter
volumeMultiplier   = input.float(1.5, "Volume Spike Multiplier", step = 0.1)

// Risk Reward
rrTarget1          = input.float(1.0, "Target 1 RR", step = 0.1)
rrTarget2          = input.float(2.0, "Target 2 RR", step = 0.1)

// Session Filters
enableSession      = input.bool(true, "Enable Session Filter")
session1           = input.session("0925-1100", "Morning Session")
session2           = input.session("1345-1445", "Afternoon Session")

// Multi Timeframe
htf                = input.timeframe("15", "Higher Timeframe")

// Optional Supertrend
useSupertrend      = input.bool(true, "Use Supertrend Filter")
stAtrPeriod        = input.int(10, "Supertrend ATR")
stFactor           = input.float(3.0, "Supertrend Factor")

// =====================================================
// CORE INDICATORS
// =====================================================

ema20      = ta.ema(close, emaLength)
vwapValue  = ta.vwap(close)
atrValue   = ta.atr(atrLength)

avgVolume  = ta.sma(volume, 20)
volumeSpike = volume > (avgVolume * volumeMultiplier)

// =====================================================
// HIGHER TIMEFRAME TREND
// =====================================================

htfClose = request.security(
     syminfo.tickerid,
     htf,
     close,
     lookahead = barmerge.lookahead_off
)

htfEMA = request.security(
     syminfo.tickerid,
     htf,
     ta.ema(close, emaLength),
     lookahead = barmerge.lookahead_off
)

htfBullish = htfClose > htfEMA
htfBearish = htfClose < htfEMA

// =====================================================
// SUPERTREND
// =====================================================

[supertrend, direction] = ta.supertrend(stFactor, stAtrPeriod)

bullishST = direction < 0
bearishST = direction > 0

// =====================================================
// MARKET STRUCTURE
// =====================================================

higherHighs = high > high[1]
higherLows  = low > low[1]

lowerHighs = high < high[1]
lowerLows  = low < low[1]

// =====================================================
// SESSION FILTER
// =====================================================

inSession1 = not na(time(timeframe.period, session1))
inSession2 = not na(time(timeframe.period, session2))

sessionAllowed = enableSession ? (inSession1 or inSession2) : true

// =====================================================
// LIQUIDITY SWEEP DETECTION
// =====================================================

// Bullish Sweep
bullSweep = low < low[1] and close > low[1]

// Bearish Sweep
bearSweep = high > high[1] and close < high[1]

// =====================================================
// VWAP RECLAIM CONDITIONS
// =====================================================

// Bullish reclaim
bullReclaim =
     close > vwapValue and
     open < close and
     close > ema20

// Bearish reclaim
bearReclaim =
     close < vwapValue and
     open > close and
     close < ema20

// =====================================================
// TREND FILTERS
// =====================================================

bullTrend =
     close > ema20 and
     ema20 > vwapValue and
     higherHighs and
     higherLows and
     htfBullish

bearTrend =
     close < ema20 and
     ema20 < vwapValue and
     lowerHighs and
     lowerLows and
     htfBearish

// =====================================================
// FINAL ENTRY CONDITIONS
// =====================================================

longCondition =
     bullTrend and
     bullSweep and
     bullReclaim and
     volumeSpike and
     sessionAllowed and
     (useSupertrend ? bullishST : true)

shortCondition =
     bearTrend and
     bearSweep and
     bearReclaim and
     volumeSpike and
     sessionAllowed and
     (useSupertrend ? bearishST : true)

// =====================================================
// ENTRY EXECUTION
// =====================================================

longSL =
     math.min(low, low[1]) - (atrValue * atrSLMultiplier)

shortSL =
     math.max(high, high[1]) + (atrValue * atrSLMultiplier)

longRisk = close - longSL
shortRisk = shortSL - close

longTP1 = close + (longRisk * rrTarget1)
longTP2 = close + (longRisk * rrTarget2)

shortTP1 = close - (shortRisk * rrTarget1)
shortTP2 = close - (shortRisk * rrTarget2)

// =====================================================
// LONG ENTRIES
// =====================================================

if (longCondition and strategy.position_size <= 0)
    strategy.entry("LONG", strategy.long)

    strategy.exit(
         "LONG TP1",
         from_entry = "LONG",
         qty_percent = 50,
         stop = longSL,
         limit = longTP1
    )

    strategy.exit(
         "LONG TP2",
         from_entry = "LONG",
         qty_percent = 100,
         stop = longSL,
         limit = longTP2
    )

// =====================================================
// SHORT ENTRIES
// =====================================================

if (shortCondition and strategy.position_size >= 0)
    strategy.entry("SHORT", strategy.short)

    strategy.exit(
         "SHORT TP1",
         from_entry = "SHORT",
         qty_percent = 50,
         stop = shortSL,
         limit = shortTP1
    )

    strategy.exit(
         "SHORT TP2",
         from_entry = "SHORT",
         qty_percent = 100,
         stop = shortSL,
         limit = shortTP2
    )

// =====================================================
// PLOTS
// =====================================================

plot(ema20, color = color.orange, linewidth = 2, title = "20 EMA")
plot(vwapValue, color = color.blue, linewidth = 2, title = "VWAP")

plot(supertrend,
     color = bullishST ? color.green : color.red,
     linewidth = 2,
     title = "Supertrend"
)

// =====================================================
// SIGNALS
// =====================================================

plotshape(
     longCondition,
     title = "BUY",
     style = shape.labelup,
     location = location.belowbar,
     color = color.green,
     text = "BUY",
     textcolor = color.white,
     size = size.small
)

plotshape(
     shortCondition,
     title = "SELL",
     style = shape.labeldown,
     location = location.abovebar,
     color = color.red,
     text = "SELL",
     textcolor = color.white,
     size = size.small
)

// =====================================================
// ALERTS
// =====================================================

alertcondition(
     longCondition,
     title = "Long Entry",
     message = "Institutional Long Setup Triggered"
)

alertcondition(
     shortCondition,
     title = "Short Entry",
     message = "Institutional Short Setup Triggered"
)

// =====================================================
// DASHBOARD
// =====================================================

var table dashboard = table.new(position.top_right, 2, 8)

if barstate.islast
    table.cell(dashboard, 0, 0, "Trend")
    table.cell(dashboard, 1, 0, bullTrend ? "Bullish" : bearTrend ? "Bearish" : "Neutral")

    table.cell(dashboard, 0, 1, "Volume Spike")
    table.cell(dashboard, 1, 1, volumeSpike ? "YES" : "NO")

    table.cell(dashboard, 0, 2, "HTF Trend")
    table.cell(dashboard, 1, 2, htfBullish ? "Bullish" : "Bearish")

    table.cell(dashboard, 0, 3, "Session")
    table.cell(dashboard, 1, 3, sessionAllowed ? "ACTIVE" : "OFF")

    table.cell(dashboard, 0, 4, "ATR")
    table.cell(dashboard, 1, 4, str.tostring(atrValue))

    table.cell(dashboard, 0, 5, "VWAP")
    table.cell(dashboard, 1, 5, str.tostring(vwapValue))

    table.cell(dashboard, 0, 6, "EMA20")
    table.cell(dashboard, 1, 6, str.tostring(ema20))

    table.cell(dashboard, 0, 7, "Strategy")
    table.cell(dashboard, 1, 7, "Institutional LS")




//@version=6
strategy(
     title = "Institutional VWAP Opening Range Strategy",
     shorttitle = "IVWAP-ORB",
     overlay = true,
     initial_capital = 500000,
     pyramiding = 0,
     commission_type = strategy.commission.percent,
     commission_value = 0.02,
     process_orders_on_close = true
)

// ======================================================
// INPUTS
// ======================================================

orbMinutes          = input.int(15, "Opening Range Minutes", minval = 5)
emaLength           = input.int(20, "EMA Length")
atrLength           = input.int(14, "ATR Length")
atrSLMultiplier     = input.float(1.0, "ATR Stop Multiplier", step = 0.1)
riskReward          = input.float(2.0, "Risk Reward Ratio", step = 0.1)
volumeMultiplier    = input.float(1.2, "Volume Spike Multiplier", step = 0.1)
useSupertrend       = input.bool(true, "Use Supertrend Filter")
stAtrLength         = input.int(10, "Supertrend ATR Length")
stFactor            = input.float(3.0, "Supertrend Factor")

// ======================================================
// INDICATORS
// ======================================================

vwapValue = ta.vwap(close)

ema20 = ta.ema(close, emaLength)

atrValue = ta.atr(atrLength)

avgVolume = ta.sma(volume, 20)

volumeSpike = volume > avgVolume * volumeMultiplier

// ======================================================
// SUPERTREND
// ======================================================

[supertrend, direction] = ta.supertrend(stFactor, stAtrLength)

bullST = direction < 0
bearST = direction > 0

// ======================================================
// OPENING RANGE LOGIC
// ======================================================

newDay = ta.change(time("D"))

var float orbHigh = na
var float orbLow = na

marketOpenHour = 9
marketOpenMinute = 15

sessionStart = timestamp(year, month, dayofmonth, marketOpenHour, marketOpenMinute)

orbEndTime = sessionStart + (orbMinutes * 60 * 1000)

insideORB = time >= sessionStart and time <= orbEndTime

if newDay
    orbHigh := na
    orbLow := na

if insideORB
    orbHigh := na(orbHigh) ? high : math.max(orbHigh, high)
    orbLow := na(orbLow) ? low : math.min(orbLow, low)

// ======================================================
// TREND CONDITIONS
// ======================================================

bullTrend =
     close > vwapValue and
     close > ema20 and
     ema20 > vwapValue

bearTrend =
     close < vwapValue and
     close < ema20 and
     ema20 < vwapValue

// ======================================================
// BREAKOUT CONDITIONS
// ======================================================

longBreakout =
     not insideORB and
     not na(orbHigh) and
     close > orbHigh and
     volumeSpike

shortBreakout =
     not insideORB and
     not na(orbLow) and
     close < orbLow and
     volumeSpike

// ======================================================
// PULLBACK ENTRY LOGIC
// ======================================================

pullbackLong =
     low <= ema20 or low <= vwapValue

pullbackShort =
     high >= ema20 or high >= vwapValue

bullishCandle =
     close > open and
     close > close[1]

bearishCandle =
     close < open and
     close < close[1]

// ======================================================
// FINAL ENTRY CONDITIONS
// ======================================================

longCondition =
     bullTrend and
     longBreakout and
     pullbackLong and
     bullishCandle and
     (useSupertrend ? bullST : true)

shortCondition =
     bearTrend and
     shortBreakout and
     pullbackShort and
     bearishCandle and
     (useSupertrend ? bearST : true)

// ======================================================
// STOP LOSS & TARGET
// ======================================================

longSL = close - (atrValue * atrSLMultiplier)
longRisk = close - longSL
longTP = close + (longRisk * riskReward)

shortSL = close + (atrValue * atrSLMultiplier)
shortRisk = shortSL - close
shortTP = close - (shortRisk * riskReward)

// ======================================================
// ENTRIES
// ======================================================

if longCondition and strategy.position_size <= 0
    strategy.entry("LONG", strategy.long)

    strategy.exit(
         "LONG EXIT",
         from_entry = "LONG",
         stop = longSL,
         limit = longTP
     )

if shortCondition and strategy.position_size >= 0
    strategy.entry("SHORT", strategy.short)

    strategy.exit(
         "SHORT EXIT",
         from_entry = "SHORT",
         stop = shortSL,
         limit = shortTP
     )

// ======================================================
// TRAILING EXIT USING EMA
// ======================================================

if strategy.position_size > 0
    if close < ema20
        strategy.close("LONG")

if strategy.position_size < 0
    if close > ema20
        strategy.close("SHORT")

// ======================================================
// PLOTS
// ======================================================

plot(vwapValue, title = "VWAP", linewidth = 2)

plot(ema20, title = "EMA 20", linewidth = 2)

plot(orbHigh, title = "ORB High", color = color.green, linewidth = 2)

plot(orbLow, title = "ORB Low", color = color.red, linewidth = 2)

plot(supertrend,
     title = "Supertrend",
     color = bullST ? color.green : color.red,
     linewidth = 2)

// ======================================================
// SIGNALS
// ======================================================

plotshape(
     longCondition,
     title = "BUY",
     style = shape.labelup,
     location = location.belowbar,
     color = color.green,
     text = "BUY"
)

plotshape(
     shortCondition,
     title = "SELL",
     style = shape.labeldown,
     location = location.abovebar,
     color = color.red,
     text = "SELL"
)

// ======================================================
// BACKGROUND TREND
// ======================================================

bgcolor(
     bullTrend ? color.new(color.green, 92) :
     bearTrend ? color.new(color.red, 92) :
     na
)

// ======================================================
// ALERTS
// ======================================================

alertcondition(longCondition, title = "Long Alert", message = "Institutional Long Setup")

alertcondition(shortCondition, title = "Short Alert", message = "Institutional Short Setup")