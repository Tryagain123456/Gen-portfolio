import os
import sys
import argparse
import uuid  # Import uuid for run IDs
import threading  # Import threading for background task
import getpass # For API keys
import chainlit as cl # Import Chainlit
from dotenv import load_dotenv # Import dotenv
from datetime import datetime, timedelta

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
import pprint # For pretty printing the final state

# --- API Key Setup ---
# Load .env file if it exists
load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# (设置您的 API 密钥)
_set_if_undefined("BYTEDANCE_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
_set_if_undefined("LANGSMITH_API_KEY")

# --- Agent Imports (Copied from your main.py) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设 main.py 在 src 目录下
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 确保 src 目录在 Python 路径中
# 假设 chainlit_app.py 与 main.py 在同一目录，或者 src 是其同级目录
# 如果 chainlit_app.py 在项目根目录，您可能需要将 'src' 添加到路径
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.agents.valuation import valuation_agent
    from src.agents.state import AgentState
    from src.agents.sentiment import sentiment_agent
    from src.agents.risk_manager import risk_management_agent
    from src.agents.technicals import technical_analyst_agent
    from src.agents.stock_forecast_agent import stock_forecast_agent
    from src.agents.portfolio_manager import portfolio_management_agent
    from src.agents.market_data import market_data_agent
    from src.agents.fundamentals import fundamentals_agent
    from src.agents.researcher_bull import researcher_bull_agent
    from src.agents.researcher_bear import researcher_bear_agent
    from src.agents.debate_room import debate_room_agent
    from src.agents.macro_analyst import macro_analyst_agent
    from src.agents.macro_news_agent import macro_news_agent
    from src.agents.intent_recognition_agent import intent_recognition_agent, chitchat_agent

    from src.utils.structured_terminal import print_structured_output
    HAS_STRUCTURED_OUTPUT = True
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Please ensure 'src' directory is in PYTHONPATH or structured correctly relative to chainlit_app.py")
    # 如果导入失败，我们不能继续，所以在这里退出或设置一个标志
    # 暂且假设导入会成功
    HAS_STRUCTURED_OUTPUT = False
    # 如果在 chainlit 运行时出现路径问题，您可能需要硬编码 'src' 路径：
    # SCRIPT_DIR = os.path.dirname(__file__)
    # sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, 'src')))
    # ... 然后重试导入 ...


# ======================================================================================
# 定义工作流 (Copied from your main.py)
# ======================================================================================

# 传入状态定义
workflow = StateGraph(AgentState)

# 添加工作流结构（定义点和边之间的关系）
workflow.add_node("intent_recognition_agent", intent_recognition_agent)
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("technical_analyst_agent", technical_analyst_agent)
workflow.add_node("stock_forecast_agent", stock_forecast_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("valuation_agent", valuation_agent)
workflow.add_node("macro_news_agent", macro_news_agent)
workflow.add_node("researcher_bull_agent", researcher_bull_agent)
workflow.add_node("researcher_bear_agent", researcher_bear_agent)
workflow.add_node("debate_room_agent", debate_room_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("macro_analyst_agent", macro_analyst_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)
workflow.add_node("chitchat_agent", chitchat_agent)

# ==================== 边定义 ====================
workflow.set_entry_point("intent_recognition_agent")

# 1. market_data_agent 获取的数据分别传递给 4 个分析 agent 和 1 个分析新闻分析 agent，进行进一步的分析
workflow.add_edge("market_data_agent", "technical_analyst_agent")
workflow.add_edge("market_data_agent", "stock_forecast_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("market_data_agent", "valuation_agent")
workflow.add_edge("market_data_agent", "macro_news_agent")
workflow.add_edge("market_data_agent", "macro_analyst_agent")

# 2. 将4个初步分析计算结果汇总后，分别传递给【多头研究员】和【空头研究员】
analyst_nodes = [
    "technical_analyst_agent",
    "stock_forecast_agent",
    "fundamentals_agent",
    "sentiment_agent",
    "valuation_agent",
]
workflow.add_edge(analyst_nodes, "researcher_bull_agent")
workflow.add_edge(analyst_nodes, "researcher_bear_agent")

# 3. 将多头和空头研究员的观点汇总后输入【辩论室】
workflow.add_edge(["researcher_bull_agent", "researcher_bear_agent"], "debate_room_agent")

# 4. 辩论时整合后依次通过【风险管理智能体】和【宏观分析智能体】进行分析
workflow.add_edge("debate_room_agent", "risk_management_agent")

# 5. 将新闻分析和宏观数据分析汇总后传给【资产组合经理】生成报告
workflow.add_edge(["risk_management_agent", "macro_news_agent", "macro_analyst_agent"], "portfolio_management_agent")

# 6. 终点为生成投资建议的【资产组合经理】
workflow.add_edge("portfolio_management_agent", END)

# 将工作流转换为可执行的程序
# 注意：我们没有像示例中那样添加 checkpointer
# 您的图是为一次性运行而设计的，而不是为多轮对话记忆而设计
app = workflow.compile()

# ======================================================================================
# Chainlit 应用程序定义
# ======================================================================================

@cl.on_chat_start
async def on_chat_start():
    """
    当新聊天会话开始时调用。
    我们在这里设置默认的投资组合。
    """
    # 模拟 main.py 中的默认值
    initial_capital = 1000000.0
    initial_position = 1000
    portfolio = {"cash": initial_capital, "stock": initial_position}
    
    # 将 portfolio 存储在用户会话中
    cl.user_session.set("portfolio", portfolio)
    
    await cl.Message(
        content="欢迎使用 Gen-Portfolio 分析助手。\n\n"
                "我已经为您设置了初始模拟投资组合：\n"
                f"- **现金:** ${initial_capital:,.2f}\n"
                f"- **持仓:** {initial_position} 股\n\n"
                "请输入您想分析的股票，例如：'我想看看万向钱潮是否值得投资'"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    处理用户发送的每条消息。
    """
    
    # 1. 创建一个消息用于显示 "正在运行" 状态
    msg = cl.Message(content="")
    await msg.send()
    
    # 2. 从会话和消息中收集运行所需的数据
    portfolio = cl.user_session.get("portfolio")
    user_input = message.content
    run_id = str(uuid.uuid4())

    # 获取当前时间（与 main.py 逻辑相同）
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date_dt = yesterday
    start_date_dt = end_date_dt - timedelta(days=365)
    
    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = end_date_dt.strftime('%Y-%m-%d')

    # 3. 构建初始状态 (与 main.py 逻辑相同)
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "data": {
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "num_of_news": 100, # 您可以硬编码或将其设为可配置
        },
        "metadata": {
            "show_reasoning": True,
            "run_id": run_id,
            "show_summary": True
        }
    }

    # 4. 运行工作流
    # app.invoke 是一个同步函数，我们使用 cl.make_async 将其转换为异步
    msg.content = "正在运行分析... 这可能需要一些时间。\n" \
                  f"分析时段: {start_date} 到 {end_date}"
    await msg.update()
    
    try:
        # 在异步函数中运行同步的 app.invoke
        final_state = await cl.make_async(app.invoke)(initial_state)
        
        # 5. 提取最终结果
        result_content = final_state.get("messages", [])[-1].get("content", "分析完成，但未找到最终报告。")

        
        # 6. 将最终报告发送给用户
        msg.content = result_content
        await msg.update()

        # 7. (可选) 发送完整的状态以供调试
        # 这模拟了您的 print_structured_output
        if HAS_STRUCTURED_OUTPUT:
            # 使用 pprint 格式化字典
            state_details = pprint.pformat(final_state, indent=2, width=120)
            await cl.Message(
                content="**完整的最终状态 (调试信息):**",
                elements=[
                    cl.Code(content=state_details, language="python", display="inline")
                ]
            ).send()

    except Exception as e:
        await cl.Message(content=f"运行分析时出错：\n{str(e)}").send()