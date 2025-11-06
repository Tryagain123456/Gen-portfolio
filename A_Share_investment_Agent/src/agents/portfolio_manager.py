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
       置信度: {(clean_confidence(fundamental_signal.get('confidence')) * 100 if fundamental_signal else 0) :.0f}%
       要点:
       - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
       - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
       - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}
       - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据') if fundamental_signal else '无数据'}

    2. 估值分析 (权重35%):
       信号: {signal_to_chinese(valuation_signal)}
       置信度: {(clean_confidence(valuation_signal.get('confidence')) * 100 if valuation_signal else 0) :.0f}%
       要点:
       - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}
       - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据') if valuation_signal else '无数据'}

    3. 技术分析 (权重25%):
       信号: {signal_to_chinese(technical_signal)}
       置信度: {(clean_confidence(technical_signal.get('confidence')) * 100 if technical_signal else 0) :.0f}%
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
          置信度: {(clean_confidence(general_macro_signal.get('confidence')) * 100 if general_macro_signal else 0) :.0f}%
          宏观环境: {general_macro_signal.get(
        'macro_environment', '无数据') if general_macro_signal else '无数据'}
          对股票影响: {general_macro_signal.get(
        'impact_on_stock', '无数据') if general_macro_signal else '无数据'}
          关键因素: {', '.join(general_macro_signal.get(
        'key_factors', ['无数据']) if general_macro_signal else ['无数据'])}

       b) 大盘宏观新闻分析 (来自 Macro News Agent):
          信号: {signal_to_chinese(market_wide_news_signal)}
          置信度: {(clean_confidence(market_wide_news_signal.get('confidence')) * 100 if market_wide_news_signal else 0) :.0f}%
          摘要或结论: {market_wide_news_signal.get(
        'reasoning', market_wide_news_summary) if market_wide_news_signal else market_wide_news_summary}

    5. 情绪分析 (权重10%):
       信号: {signal_to_chinese(sentiment_signal)}
       置信度: {(clean_confidence(sentiment_signal.get('confidence')) * 100 if sentiment_signal else 0) :.0f}%
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

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }





#"portfolio_management", "负责投资组合管理和最终交易决策")
def portfolio_management_agent(state: AgentState):
    """Responsible for portfolio management"""
    agent_name = "portfolio_management_agent"
    show_workflow_status("Portfolio Management Agent")

    unique_incoming_messages = {}
    for msg in state["messages"]:
        # Keep overriding with later messages to get the latest by name
        unique_incoming_messages[msg.name] = msg

    cleaned_messages_for_processing = list(unique_incoming_messages.values())
    # logger.info(
    # f"--- DEBUG: {agent_name} CLEANED messages for processing: {[msg.name for msg in cleaned_messages_for_processing]} ---")

    # show_workflow_status(f"{agent_name}: --- Executing Portfolio Manager ---")
    show_reasoning_flag = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get messages from other agents using the cleaned list
    technical_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "technical_analyst_agent")
    fundamentals_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "fundamentals_agent")
    sentiment_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "sentiment_agent")
    valuation_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "valuation_agent")
    risk_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "risk_management_agent")
    tool_based_macro_message = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_analyst_agent")  # This is the main analysis path output

    # Extract content, handling potential None if message not found by get_latest_message_by_name
    technical_content = technical_message.content if technical_message else json.dumps(
        {"signal": "error", "details": "Technical message missing"})
    fundamentals_content = fundamentals_message.content if fundamentals_message else json.dumps(
        {"signal": "error", "details": "Fundamentals message missing"})
    sentiment_content = sentiment_message.content if sentiment_message else json.dumps(
        {"signal": "error", "details": "Sentiment message missing"})
    valuation_content = valuation_message.content if valuation_message else json.dumps(
        {"signal": "error", "details": "Valuation message missing"})
    risk_content = risk_message.content if risk_message else json.dumps(
        {"signal": "error", "details": "Risk message missing"})
    tool_based_macro_content = tool_based_macro_message.content if tool_based_macro_message else json.dumps(
        {"signal": "error", "details": "Tool-based Macro message missing"})

    # Market-wide news summary from macro_news_agent (already correctly fetched from state["data"])
    market_wide_news_summary_content = state["data"].get(
        "macro_news_analysis_result", "大盘宏观新闻分析不可用或未提供。")
    # Optional: also try to get the message object for consistency in agent_signals, though data field is primary source
    macro_news_agent_message_obj = get_latest_message_by_name(
        cleaned_messages_for_processing, "macro_news_agent")

    system_message_content = """你是一名负责最终交易决策的投资组合经理。
你的工作是基于团队的分析做出交易决策，同时严格遵守风险管理约束。

风险管理约束：
- 你绝不能超过风险经理规定的最大持仓规模
- 你必须遵循风险管理建议的交易动作（买入/卖出/持有）
- 这些是硬性约束，不能被其他信号覆盖

在权衡不同信号的方向和时机时：
1. 估值分析（30%权重）
2. 基本面分析（25%权重）
3. 技术分析（20%权重）
   a) 技术分析信号（来自技术分析师代理）
   b) 股票价格预测（来自股票预测代理）
4. 宏观分析（15%权重）- 包含两个输入：
   a) 总体宏观环境（来自宏观分析师代理，基于工具分析）
   b) 每日市场新闻摘要（来自宏观新闻代理）
   两者都为外部风险和机会提供背景信息
5. 情绪分析（10%权重）
决策流程应为：
1. 首先检查风险管理约束
2. 然后评估估值信号
3. 接着评估基本面信号
4. 同时考虑总体宏观环境和每日市场新闻摘要
5. 利用技术分析确定时机
6. 考虑情绪因素进行最终调整

请在输出的JSON中提供以下内容：
- "action": "buy" | "sell" | "hold"（买入|卖出|持有）
- "quantity": <正整数>（数量）
- "confidence": <0到1之间的浮点数>（置信度）
- "reasoning": <决策的简要解释，包括如何权衡所有信号，包括两个宏观输入>

交易规则：
- 绝不超过风险管理的持仓限制
- 只有在有可用现金时才能买入
- 只有在有股票可卖时才能卖出
- 卖出数量必须≤当前持仓量
- 数量必须≤风险管理规定的最大持仓规模

输出示例：
{"action": "hold", "quantity": 0, "confidence": 0.75, "reasoning": "决策遵循风险管理明确建议的'hold'操作。技术分析(20%权重)信号中性且置信度极低(0%)，基本面分析(25%权重)信号中性(25%置信度)，显示盈利能力看跌但财务健康看涨相互抵消。估值分析(30%权重)信号中性且置信度无法确定。情绪分析(10%权重)信号中性(100%置信度)。宏观分析中，特定股票宏观分析显示管理层变动带来负面不确定性(60%置信度)，而市场-wide新闻摘要显示整体谨慎乐观、结构分化但未提供明确方向性指引(50%置信度)。综合所有中性或相互抵消的信号，且风险管理建议持有，因此决定维持现有持仓状态。"}
"""
    system_message = {
        "role": "system",
        "content": system_message_content
    }

    agent_signals = {
        "technical_signal": parse_json_signal(technical_content),
        "fundamental_signal": parse_json_signal(fundamentals_content),
        "sentiment_signal": parse_json_signal(sentiment_content),
        "valuation_signal": parse_json_signal(valuation_content),
        "risk_signal": parse_json_signal(risk_content),
        "general_macro_signal": parse_json_signal(tool_based_macro_content),
        "market_wide_news_signal": parse_json_signal(market_wide_news_summary_content),
        "portfolio_cash": round(portfolio["cash"], 2),
        "portfolio_stock_shares": portfolio["stock"],
    }

    print(f"agent_signals:\n {agent_signals}\n\n")
    user_message_content = f"""基于下面的团队分析结果, 做出投资决策.
    
            Technical Analysis Signal: {technical_content}
            Stock Price Forecast: {state["data"].get("predicted_price_data", "未提供股票价格预测结果")}
            Fundamental Analysis Signal: {fundamentals_content}
            Sentiment Analysis Signal: {sentiment_content}
            Valuation Analysis Signal: {valuation_content}
            Risk Management Signal: {risk_content}
            General Macro Analysis (from Macro Analyst Agent): {tool_based_macro_content}
            Daily Market-Wide News Summary (from Macro News Agent):
            {market_wide_news_summary_content}

            Current Portfolio:
            Cash: {portfolio['cash']:.2f}
            Current Position: {portfolio['stock']} shares

            Output JSON only. Ensure 'agent_signals' includes all required agents as per system prompt."""
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    # show_agent_reasoning(agent_name, f"Preparing LLM. User msg includes: TA, FA, Sent, Val, Risk, GeneralMacro, MarketNews.")

    llm_interaction_messages = [system_message, user_message]
    llm_response_content = get_chat_completion(llm_interaction_messages)

    current_metadata = state["metadata"]
    current_metadata["current_agent_name"] = agent_name


    if llm_response_content is None:
        # Ensure the dummy response matches the expected structure for agent_signals
        llm_response_content = json.dumps({
            "action": "hold",
            "quantity": 0,
            "confidence": 0.7,
            "reasoning": "在调用大语言模型（LLM）API 过程中发生了故障，系统已自动切换到基于保守的持仓策略"
        })

    if show_reasoning_flag:
        show_agent_reasoning(
            agent_name, f"Final LLM decision JSON: {llm_response_content}")

    agent_decision_details_value = {}
    final_report_content = ""
    try:

        decision_json = json.loads(llm_response_content) # type: ignore

        reasoning_text = decision_json.get("reasoning") or ""  # "or" 会处理 None 和 ""
        agent_decision_details_value = {
            "action": decision_json.get("action"),
            "quantity": decision_json.get("quantity"),
            "confidence": decision_json.get("confidence"),
            # "reasoning_snippet": reasoning_text + "..."
            "reasoning_snippet": reasoning_text
        }
        print("="*50)
        print(f"用于报告生成的数据 decision_json：{decision_json}")
        print("=" * 50)
        # 2. 调用 format_decision 来生成漂亮的报告

        formatted_result = format_decision(
            action=decision_json.get("action", "hold"),
            quantity=decision_json.get("quantity", 0),
            confidence=decision_json.get("confidence", 0.0),
            # agent_signals=decision_json.get("agent_signals", []),
            agent_signals = agent_signals,
            reasoning=decision_json.get("reasoning", "无决策依据。"),
            market_wide_news_summary=state["data"].get(
                "macro_news_analysis_result", "大盘宏观新闻分析不可用。")
        )
        print("=" * 50+"\n\n")
        print(f"用于格式化的数据:{formatted_result}")
        print("=" * 50)
        # 3. 将格式化后的报告 (字符串) 作为最终的消息内容
        final_report_content = formatted_result.get("分析报告", llm_response_content)

    except Exception as e:

        logger.error(f"无法解析或处理 portfolio_manager 的 LLM 响应: {e}")
        agent_decision_details_value = {
            "error": f"处理 LLM 决策时出错: {e}",
            "raw_response_snippet": llm_response_content[:200] + "..." # type: ignore
        }
        final_report_content = f"LLM 响应处理失败 (错误: {e})：\n{llm_response_content}"

    # show_workflow_status(f"{agent_name}: --- Portfolio Manager Completed ---")
    final_decision_message = HumanMessage(
        content=final_report_content,
    )

    # show_workflow_status(f"{agent_name}: --- Portfolio Manager Completed ---")

    final_messages_output = cleaned_messages_for_processing + [final_decision_message]

    # logger.info(f"--- DEBUG: {agent_name} RETURN messages: {[msg.name for msg in final_messages_output]} ---")

    serializable_messages = [message_to_dict(m) for m in final_messages_output]
    return_payload = {
        "messages": serializable_messages,
        "data": state["data"],
        "metadata": {
            **state["metadata"],
            f"{agent_name}_decision_details": agent_decision_details_value,
            "agent_reasoning": llm_response_content
        }
    }
    # logger.info(f"--- DEBUG: {agent_name} Cleaning payload for serialization... ---")
    cleaned_payload = convert_numpy_types(return_payload)

    return cleaned_payload


