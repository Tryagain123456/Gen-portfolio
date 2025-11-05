import os
import sys
import argparse
import uuid  # Import uuid for run IDs
import threading  # Import threading for background task
# import uvicorn  # Import uvicorn to run FastAPI

from datetime import datetime, timedelta
# Removed START as it's implicit with set_entry_point
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
# import pandas as pd
# import akshare as ak

# --- Agent Imports ---
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设 main.py 在 src 目录下
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.agents.valuation import valuation_agent
from src.agents.state import AgentState
from src.agents.sentiment import sentiment_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.market_data import market_data_agent
from src.agents.fundamentals import fundamentals_agent
from src.agents.researcher_bull import researcher_bull_agent
from src.agents.researcher_bear import researcher_bear_agent
from src.agents.debate_room import debate_room_agent
from src.agents.macro_analyst import macro_analyst_agent
from src.agents.macro_news_agent import macro_news_agent
from src.agents.intent_recognition_agent import intent_recognition_agent

try:
    from src.utils.structured_terminal import print_structured_output
    HAS_STRUCTURED_OUTPUT = True
except ImportError:
    HAS_STRUCTURED_OUTPUT = False

# ======================================================================================

# 工作流运行函数

def run_hedge_fund(
        run_id: str,
        user_input: str,
        start_date: str,
        end_date: str,
        portfolio: dict):
    print(f"--- 开始投资策略分析 Run ID: {run_id} ---")

    messages = [HumanMessage(content=user_input)]

    initial_state = {
        "messages": messages,
        "data": {
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "num_of_news": 100,
        },
        "metadata": {
            "show_reasoning": True,
            "run_id": run_id,
            "show_summary": True
        }
    }


    final_state = app.invoke(initial_state) # 将初始状态传入工作流，触发整个流程运行，返回运行涉及的所有信息
    print(f"--- 投资策略分析完成 Run ID: {run_id} ---")

    if HAS_STRUCTURED_OUTPUT:
        print_structured_output(final_state)

    return final_state["messages"][-1]["content"]

# ======================================================================================

# 定义工作流
## 传入状态定义
workflow = StateGraph(AgentState)
## 添加工作流结构（定义点和边之间的关系）
workflow.add_node("intent_recognition_agent", intent_recognition_agent)
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("technical_analyst_agent", technical_analyst_agent)
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


# ==================== 边定义 ====================

# workflow.set_entry_point("market_data_agent")

workflow.set_entry_point("intent_recognition_agent")
workflow.add_edge("intent_recognition_agent", "market_data_agent")

# workflow.add_conditional_edges(
#     "intent_recognition_agent",
#     route_intent,
#     {
#         "start_analysis": "market_data_agent", # 路由到分析流程
#         "chit_chat": "small_talk_agent",       # <--- 新路由: 到闲聊
#         "wait_for_user_input": END             # 路由到结束 (等待用户)
#     }
# )
# workflow.add_edge("small_talk_agent", END)

# 1. market_data_agent 获取的数据分别传递给 4 个分析 agent 和 1 个分析新闻分析 agent，进行进一步的分析
workflow.add_edge("market_data_agent", "technical_analyst_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("market_data_agent", "valuation_agent")
workflow.add_edge("market_data_agent", "macro_news_agent")


# 2. 将4个初步分析计算结果汇总后，分别传递给【多头研究员】和【空头研究员】
analyst_nodes = [
    "technical_analyst_agent",
    "fundamentals_agent",
    "sentiment_agent",
    "valuation_agent"
]
workflow.add_edge(analyst_nodes, "researcher_bull_agent")
workflow.add_edge(analyst_nodes, "researcher_bear_agent")

# 3. 将多头和空头研究员的观点汇总后输入【辩论室】
workflow.add_edge(["researcher_bull_agent", "researcher_bear_agent"], "debate_room_agent")

# 4. 辩论时整合后依次通过【风险管理智能体】和【宏观分析智能体】进行分析
workflow.add_edge("debate_room_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "macro_analyst_agent")

# 5. 将新闻分析和宏观数据分析汇总后传给【资产组合经理】生成报告
workflow.add_edge(["macro_analyst_agent", "macro_news_agent"], "portfolio_management_agent")

# 6. 终点为生成投资建议的【资产组合经理】
workflow.add_edge("portfolio_management_agent", END)

# 将工作流转换为可执行的程序
app = workflow.compile()
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
# ======================================================================================

import getpass
import uuid
# --- Main Execution Block ---
if __name__ == "__main__":
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"Please provide your {var}")


    # (设置您的 API 密钥)
    _set_if_undefined("BYTEDANCE_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")
    _set_if_undefined("LANGSMITH_API_KEY")
    class Args:
        def __init__(self):
            self.user_input = "我想看看宁德时代是否值得投资"
            self.initial_capital = 1000000.0
            self.initial_position = 1000
    args = Args()

    # 获取当前时间（分析基于前一天时间进行，保证end_data的数据具有完整性，start_data默认为一年前）
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday
    start_date = end_date - timedelta(days=365)

    # 初始的投资组合状态（资金 & 持仓数量）
    portfolio = {"cash": args.initial_capital, "stock": args.initial_position}
    # 运行函数
    result = run_hedge_fund(
        run_id = str(uuid.uuid4()),
        user_input = args.user_input,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        portfolio=portfolio
    )
    print("\nFinal Result:")
    print(result)
