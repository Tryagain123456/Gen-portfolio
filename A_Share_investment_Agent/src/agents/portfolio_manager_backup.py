
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from src.utils.logging_config import setup_logger

from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.openrouter_config import get_chat_completion

# 初始化 logger
logger = setup_logger('portfolio_management_agent')

# (可以放在 portfolio_manager.py 的顶部)
import numpy as np
import collections.abc  # 用于更稳健的 dict/list 检查


def convert_numpy_types(data):
    """
    递归地将所有 numpy 类型、不可序列化对象转换为 Python 原生类型。
    支持 dict、list、tuple、set、LangChain Message 等。
    """
    import numpy as np
    import collections.abc

    if isinstance(data, collections.abc.Mapping):
        return {k: convert_numpy_types(v) for k, v in data.items()}

    elif isinstance(data, (list, tuple, set)):
        # tuple/set也处理掉
        return [convert_numpy_types(v) for v in data]

    elif isinstance(data, np.generic):
        # ✅ 统一转换所有 numpy 标量（float64, int64, bool_等）
        return data.item()

    elif isinstance(data, np.ndarray):
        return [convert_numpy_types(x) for x in data.tolist()]

    elif hasattr(data, "__dict__"):
        # ✅ 处理 LangChain Message、Pydantic、dataclass 等
        return convert_numpy_types(vars(data))

    elif hasattr(data, "_asdict"):
        # ✅ namedtuple
        return convert_numpy_types(data._asdict())

    elif isinstance(data, (float, int, str, bool)) or data is None:
        return data

    else:
        try:
            json.dumps(data)
            return data
        except Exception:
            return str(data)

##### Portfolio Management Agent #####
def get_latest_message_by_name(messages: list, name: str):
    for msg in reversed(messages):
        if msg.name == name:
            return msg
    logger.warning(
        f"Message from agent '{name}' not found in portfolio_management_agent.")
    # Return a dummy message object or raise an error, depending on desired handling
    # For now, returning a dummy message to avoid crashing, but content will be None.
    return HumanMessage(content=json.dumps({"signal": "error", "details": f"Message from {name} not found"}), name=name)

def message_to_dict(msg):
    """将 LangChain 的 HumanMessage/AIMessage 转换为可序列化字典"""
    if hasattr(msg, "content"):
        return {
            "type": msg.__class__.__name__,
            "name": getattr(msg, "name", None),
            "content": msg.content,
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        }
    elif isinstance(msg, dict):
        return msg
    return str(msg)

def parse_json_signal(signal_str):
    if not signal_str:
        return {}  # 空字符串返回空字典
    try:
        return json.loads(signal_str)  # 解析 JSON 字符串为字典
    except json.JSONDecodeError:
        return {}  # 解析失败也返回空字典


def clean_confidence(confidence_str_or_num):
    """将 '19%' 或 0.19 或 '1928%' 这样的值统一转为 0.0 到 1.0 之间的浮点数"""
    if isinstance(confidence_str_or_num, (int, float)):
        # 如果是数字 1928 或 19，我们假设它是百分比
        if confidence_str_or_num > 1.0:
            return confidence_str_or_num / 100.0
        return confidence_str_or_num  # 已经是 0.19 这样的浮点数

    if isinstance(confidence_str_or_num, str):
        try:
            # 移除 '%' 并转换为浮点数，然后除以 100
            return float(confidence_str_or_num.replace('%', '')) / 100.0
        except ValueError:
            # 如果转换失败（比如字符串是空的或无效的）
            return 0.0

    return 0.0  # 默认




def format_decision(action: str, quantity: int, confidence: float, agent_signals: dict, reasoning: str,
                    market_wide_news_summary: str = "未提供") -> dict:
    """Format the trading decision into a standardized output format.
    Think in English but output analysis in Chinese."""

    # fundamental_signal = next((s for s in agent_signals if s and s.get("agent") == "fundamental_analysis"), None)
    # valuation_signal = next((s for s in agent_signals if s and s.get("agent") == "valuation_analysis"), None)
    # technical_signal = next((s for s in agent_signals if s and s.get("agent") == "technical_analysis"), None)
    # sentiment_signal = next((s for s in agent_signals if s and s.get("agent") == "sentiment_analysis"), None)
    # risk_signal = next((s for s in agent_signals if s and s.get("agent") == "risk_management"), None)
    # general_macro_signal = next((s for s in agent_signals if s and s.get("agent") == "selected_stock_macro_analysis"),
    #                             None)
    # market_wide_news_signal = next((s for s in agent_signals if s and s.get("agent") == "market_wide_news_summary"),
    #                                None)

    fundamental_signal = agent_signals.get("fundamental_signal")
    valuation_signal = agent_signals.get("valuation_signal")
    technical_signal =  agent_signals.get("technical_signal")
    sentiment_signal =  agent_signals.get("sentiment_signal")
    risk_signal = agent_signals.get("risk_signal")
    general_macro_signal = agent_signals.get("general_macro_signal")
    market_wide_news_signal = agent_signals.get("market_wide_news_signal")

    def signal_to_chinese(signal_data):
        if not signal_data:
            return "无数据"
        if signal_data.get("signal") == "bullish":
            return "看多"
        if signal_data.get("signal") == "bearish":
            return "看空"
        return "中性"




    detailed_analysis = f"""
====================================
          投资分析报告
====================================

一、策略分析

1. 基本面分析 (权重30%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {(clean_confidence(fundamental_signal.get('confidence')) * 100) :.0f}%
   要点:
   - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
   - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}

2. 估值分析 (权重35%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {(clean_confidence(valuation_signal.get('confidence')) * 100) :.0f}%
   要点:
   - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
   - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}

3. 技术分析 (权重25%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {(clean_confidence(technical_signal.get('confidence')) * 100) :.0f}%
   要点:
   - 趋势跟踪: ADX={(technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', 0.0) if technical_signal else 0.0) :.2f}
   - 均值回归: RSI(14)={(technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', 0.0) if technical_signal else 0.0) :.2f}
   - 动量指标:
     * 1月动量={(technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', 0.0) if technical_signal else 0.0) :.2%}
     * 3月动量={(technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', 0.0) if technical_signal else 0.0) :.2%}
     * 6月动量={(technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', 0.0) if technical_signal else 0.0) :.2%}
   - 波动性: {(technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', 0.0) if technical_signal else 0.0) :.2%}

4. 宏观分析 (综合权重15%):
   a) 常规宏观分析 (来自 Macro Analyst Agent):
      信号: {signal_to_chinese(general_macro_signal)}
      置信度: {(general_macro_signal.get('confidence', 0) * 100 if general_macro_signal else 0) :.0f}%
      宏观环境: {general_macro_signal.get(
        'macro_environment', '无数据') if general_macro_signal else '无数据'}
      对股票影响: {general_macro_signal.get(
        'impact_on_stock', '无数据') if general_macro_signal else '无数据'}
      关键因素: {', '.join(general_macro_signal.get(
        'key_factors', ['无数据']) if general_macro_signal else ['无数据'])}

   b) 大盘宏观新闻分析 (来自 Macro News Agent):
      信号: {signal_to_chinese(market_wide_news_signal)}
      置信度: {(market_wide_news_signal.get('confidence', 0) * 100 if market_wide_news_signal else 0) :.0f}%
      摘要或结论: {market_wide_news_signal.get(
        'reasoning', market_wide_news_summary) if market_wide_news_signal else market_wide_news_summary}

5. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {(sentiment_signal.get('confidence', 0) * 100 if sentiment_signal else 0) :.0f}%
   分析: {sentiment_signal.get('reasoning', '无详细分析')
    if sentiment_signal else '无详细分析'}

二、风险评估
 risks_score: {risk_signal.get('risk_score', '无数据') if risk_signal else '无数据'}/10
主要指标:
- 波动率: {(risk_signal.get('risk_metrics', {}).get('volatility', 0.0) * 100 if risk_signal else 0.0) :.1f}%
- 最大回撤: {(risk_signal.get('risk_metrics', {}).get('max_drawdown', 0.0) * 100 if risk_signal else 0.0) :.1f}%
- VaR(95%): {(risk_signal.get('risk_metrics', {}).get('value_at_risk_95', 0.0) * 100 if risk_signal else 0.0) :.1f}%
- 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据') if risk_signal else '无数据'}/10

三、投资建议
操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
交易数量: {quantity}股
决策置信度: {confidence * 100:.0f}%

四、决策依据
{reasoning}

===================================="""

    show_workflow_status("Portfolio Management Agent", "completed")

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }




agent_signals={'technical_signal': {'signal': 'neutral', 'confidence': '19%', 'strategy_signals': {'trend_following': {'signal': 'bullish', 'confidence': '28%', 'metrics': {'adx': 27.519745774149236, 'trend_strength': 0.27519745774149235}}, 'mean_reversion': {'signal': 'neutral', 'confidence': '50%', 'metrics': {'z_score': 0.7262499895746909, 'price_vs_bb': 0.4592125736845533, 'rsi_14': 56.0199625701809, 'rsi_28': 50.41065482796891}}, 'momentum': {'signal': 'neutral', 'confidence': '50%', 'metrics': {'momentum_1m': 0.004019618280787229, 'momentum_3m': 0.344459859798405, 'momentum_6m': 0.5209318384811678, 'volume_momentum': 0.6856700830980349}}, 'volatility': {'signal': 'neutral', 'confidence': '50%', 'metrics': {'historical_volatility': 0.4419393053258721, 'volatility_regime': 0.8424967752099451, 'volatility_z_score': -1.3005798716065493, 'atr_ratio': 0.03593387744331143}}, 'statistical_arbitrage': {'signal': 'neutral', 'confidence': '50%', 'metrics': {'hurst_exponent': 4.410881380903355e-15, 'skewness': 0.766886377348506, 'kurtosis': 1.3585106116706094}}}}, 'fundamental_signal': {'signal': 'bullish', 'confidence': '100%', 'reasoning': {'profitability_signal': {'signal': 'bullish', 'details': 'ROE: 1560.00%, Net Margin: 1847.48%, Op Margin: 2139.11%'}, 'growth_signal': {'signal': 'bullish', 'details': 'Revenue Growth: 927.53%, Earnings Growth: 3501.80%'}, 'financial_health_signal': {'signal': 'bullish', 'details': 'Current Ratio: 1.68, D/E: 61.27'}, 'price_ratios_signal': {'signal': 'bullish', 'details': 'P/E: 17.89, P/B: 3.13, P/S: 0.28'}}}, 'sentiment_signal': {'signal': 'bullish', 'confidence': '80%', 'reasoning': 'Based on 42 recent news articles, sentiment score: 0.80'}, 'valuation_signal': {'signal': 'bullish', 'confidence': '1928%', 'reasoning': {'dcf_analysis': {'signal': 'bullish', 'details': 'Intrinsic Value: $2,850,510,117,959.78, Market Cap: $78,642,219,400.00, Gap: 3524.7%'}, 'owner_earnings_analysis': {'signal': 'bullish', 'details': 'Owner Earnings Value: $339,970,485,052.25, Market Cap: $78,642,219,400.00, Gap: 332.3%'}}}, 'risk_signal': {'max_position_size': 258553.125, 'risk_score': 3, 'trading_action': 'buy', 'risk_metrics': {'volatility': 0.37903007100659514, 'value_at_risk_95': -0.029148852594346273, 'max_drawdown': -0.23762082885819047, 'market_risk_score': 3, 'stress_test_results': {'market_crash': {'potential_loss': -75790.0, 'portfolio_impact': -0.05496210885093731}, 'moderate_decline': {'potential_loss': -37895.0, 'portfolio_impact': -0.027481054425468655}, 'slight_decline': {'potential_loss': -18947.5, 'portfolio_impact': -0.013740527212734327}}}, 'debate_analysis': {'bull_confidence': 5.345000000000001, 'bear_confidence': 0.3, 'debate_confidence': 5.345000000000001, 'debate_signal': 'bullish'}, 'reasoning': 'Risk Score 3/10: Market Risk=3, Volatility=37.90%, VaR=-2.91%, Max Drawdown=-23.76%, Debate Signal=bullish'}, 'general_macro_signal': {'macro_environment': 'neutral', 'impact_on_stock': 'neutral', 'key_factors': [], 'reasoning': 'LLM未返回有效的JSON格式结果'}, 'market_wide_news_signal': {}, 'portfolio_cash': 1000000.0, 'portfolio_stock_shares': 1000}
decision_json={'action': 'buy', 'quantity': 257553, 'confidence': 0.85, 'reasoning': "决策遵循风险管理明确建议的'buy'操作。估值分析(30%权重)信号强烈看涨且置信度极高(1928%置信度)，显示内在价值与市场价值存在3524.7%的巨大差距，所有者收益分析也表明股票被严重低估。基本面分析(25%权重)信号强烈看涨(100%置信度)，盈利能力、增长指标和财务健康状况均显示强劲正面信号。技术分析(20%权重)信号中性(19%置信度)，其中趋势跟踪策略给出弱看涨信号(28%置信度)，其他技术策略信号中性。宏观分析中，总体宏观环境为中性且对股票影响中性，而市场-wide新闻摘要显示整体市场情绪谨慎，呈现指数调整但结构分化特征，防御性板块活跃。情绪分析(10%权重)信号强烈看涨(80%置信度)，基于42篇新闻文章的综合情绪得分为0.80。风险管理建议明确买入，风险评分为3/10属于可接受范围，最大持仓规模为258553.125股，当前持仓1000股，因此决定买入257553股以达到最大允许持仓规模。"}


formatted_result = format_decision(
    action=decision_json.get("action", "hold"),
    quantity=decision_json.get("quantity", 0),
    confidence=decision_json.get("confidence", 0.0),
    # agent_signals=decision_json.get("agent_signals", []),
    agent_signals=agent_signals,
    reasoning=decision_json.get("reasoning", "无决策依据。"),
)
from pprint import pprint
pprint(formatted_result)



