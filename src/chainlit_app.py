import os
import sys

from pathlib import Path
import uuid  # Import uuid for run IDs
import traceback
import getpass # For API keys
import chainlit as cl # Import Chainlit
from playwright.async_api import async_playwright
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
_set_if_undefined("LANGSMITH_API_KEY")

# --- Agent Imports (Copied from your main.py) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 假设 main.py 在 src 目录下
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.agents.valuation import valuation_analysis_tool
    from src.agents.state import AgentState
    from src.agents.online_sentiment import online_sentiment_agent
    from src.agents.risk_assessment import risk_assessment_tool
    from src.agents.technicals import technical_analysis_tool
    from src.agents.stock_forecast import stock_forecast_tool
    from src.agents.summary_synthesis import summary_synthesis_agent
    from src.agents.market_data import market_data_tool
    from src.agents.fundamentals import fundamentals_analysis_tool
    from src.agents.bullish_research import bullish_research_agent
    from src.agents.bearish_research import bearish_research_agent
    from src.agents.tripartite_judgment import tripartite_judgment_agent
    from src.agents.macro_market import macro_market_agent
    from src.agents.macro_news import macro_news_agent
    from src.agents.intent_recognition import intent_recognition_agent, chitchat_agent

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
workflow.add_node("market_data_tool", market_data_tool)
workflow.add_node("technical_analysis_tool", technical_analysis_tool)
workflow.add_node("stock_forecast_tool", stock_forecast_tool)
workflow.add_node("fundamentals_analysis_tool", fundamentals_analysis_tool)
workflow.add_node("online_sentiment_agent", online_sentiment_agent)
workflow.add_node("valuation_analysis_tool", valuation_analysis_tool)
workflow.add_node("macro_news_agent", macro_news_agent)
workflow.add_node("bullish_research_agent", bullish_research_agent)
workflow.add_node("bearish_research_agent", bearish_research_agent)
workflow.add_node("tripartite_judgment_agent", tripartite_judgment_agent)
workflow.add_node("risk_assessment_tool", risk_assessment_tool)
workflow.add_node("macro_market_agent", macro_market_agent)
workflow.add_node("summary_synthesis_agent", summary_synthesis_agent)
workflow.add_node("chitchat_agent", chitchat_agent)

# ==================== 边定义 ====================
workflow.set_entry_point("intent_recognition_agent")

# 1. market_data_tool 获取的数据分别传递给 4 个分析 agent 和 1 个分析新闻分析 agent，进行进一步的分析
workflow.add_edge("market_data_tool", "technical_analysis_tool")
workflow.add_edge("market_data_tool", "stock_forecast_tool")
workflow.add_edge("market_data_tool", "fundamentals_analysis_tool")
workflow.add_edge("market_data_tool", "online_sentiment_agent")
workflow.add_edge("market_data_tool", "valuation_analysis_tool")
workflow.add_edge("market_data_tool", "macro_news_agent")
workflow.add_edge("market_data_tool", "macro_market_agent")

# 2. 将4个初步分析计算结果汇总后，分别传递给【多头研究员】和【空头研究员】
analyst_nodes = [
    "technical_analysis_tool",
    "stock_forecast_tool",
    "fundamentals_analysis_tool",
    "online_sentiment_agent",
    "valuation_analysis_tool",
]
workflow.add_edge(analyst_nodes, "bullish_research_agent")
workflow.add_edge(analyst_nodes, "bearish_research_agent")

# 3. 将多头和空头研究员的观点汇总后输入【辩论室】
workflow.add_edge(["bullish_research_agent", "bearish_research_agent"], "tripartite_judgment_agent")

# 4. 辩论时整合后依次通过【风险管理智能体】和【宏观分析智能体】进行分析
workflow.add_edge("tripartite_judgment_agent", "risk_assessment_tool")

# 5. 将新闻分析和宏观数据分析汇总后传给【资产组合经理】生成报告
workflow.add_edge(["risk_assessment_tool", "macro_news_agent", "macro_market_agent"], "summary_synthesis_agent")

# 6. 终点为生成投资建议的【资产组合经理】
workflow.add_edge("summary_synthesis_agent", END)

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
    
    await cl.Message(
        content="欢迎使用 Gen-Portfolio 分析助手。\n\n"
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
    user_input = message.content
    run_id = str(uuid.uuid4())

    # 获取当前时间（与 main.py 逻辑相同）
    now_dt = datetime.now()
    yesterday = now_dt - timedelta(days=1)
    end_date_dt = yesterday
    start_date_dt = end_date_dt - timedelta(days=365)

    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = end_date_dt.strftime('%Y-%m-%d')

    # 3. 构建初始状态 (与 main.py 逻辑相同)
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "data": {
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

    # 4. 运行工作流
    msg.content = "正在运行分析... 这可能需要一些时间。\n" \
                  f"分析时段: {start_date} 到 {end_date}"
    await msg.update()

    try:
        # 在异步函数中运行同步的 app.invoke（保持你原来的调用方式）
        final_state = await cl.make_async(app.invoke)(initial_state)

        # 5. 提取最终结果
        result_content = final_state.get("messages", [])[-1].get("content", "分析完成，但未找到最终报告。")

        # 6. 提取股票预测结果图（以脚本目录为基准）
        stock_ticker = final_state.get("data", {}).get("ticker", "")
        current_date_str = datetime.now().strftime("%Y%m%d")

        # 明确以脚本文件所在目录为基准（更稳妥）
        base_dir = Path(__file__).parent.parent
        file_path = (base_dir / "output_images_kronos" / f"{stock_ticker}_{current_date_str}_pred_90d.html").resolve()
        png_path = file_path.with_suffix(".png")

        # 确保输出目录存在
        png_path.parent.mkdir(parents=True, exist_ok=True)

        # 7. 使用 Playwright 渲染并截图（推荐）
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)  # 若在容器中可能加 args=['--no-sandbox']
                page = await browser.new_page()

                # 设置视窗大小（按需要调整）
                await page.set_viewport_size({"width": 1200, "height": 800})

                # 使用 file:// URI 打开页面并等待网络空闲
                file_uri = file_path.as_uri()
                await page.goto(file_uri, wait_until="networkidle")

                # 截图
                await page.screenshot(path=str(png_path), full_page=True)

            try:
                img_element = cl.Image(path=str(png_path))
                await cl.Message(content=result_content + "\n\n #### 📊 以下是该股票未来90日的预测图：\n\n", elements=[img_element]).send()
            except Exception as e:
                await cl.Message(content=f"cl.Image(path=...) 股票未来90日的预测图发送失败：{e}\n```\n{traceback.format_exc()}\n```").send()


        except Exception as e_render:
            tb = traceback.format_exc()
            msg.content = result_content + f"\n\n⚠️ 股票未来90日的预测图渲染或发送出错：{e_render}\n```\n{tb}\n```"
            await msg.update()
            return

        # 9. (可选) 发送完整的状态以供调试
        if HAS_STRUCTURED_OUTPUT:
            state_details = pprint.pformat(final_state, indent=2, width=120)
            await cl.Message(
                content="**完整的最终状态 (调试信息):**",
                elements=[cl.Code(content=state_details, language="python", display="inline")]
            ).send()

    except Exception as e:
        tb = traceback.format_exc()
        await cl.Message(content=f"运行分析时出错：\n{e}\n```\n{tb}\n```").send()
