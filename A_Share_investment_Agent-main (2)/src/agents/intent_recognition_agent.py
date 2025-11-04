import sys, os
from datetime import datetime, timedelta

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
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from typing import Optional, Literal
from model.create_model import create_model



class AnalysisParameters(BaseModel):
    """用于股票分析的参数。"""
    ticker: Optional[str] = Field(
        None,
        description="根据公司名称**推断**出的股票代码。例如：'贵州茅台' -> '600519', '大智慧' -> '601519'。请使用你内部的知识库来完成推断。")
    company_name: Optional[str] = Field(
        None,
        description="用户提到的公司名称, 例如 '苹果' 或 '大智慧'。")
    start_date: Optional[str] = Field(None, description="分析的开始日期 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="分析的结束日期 (YYYY-MM-DD)")
    initial_capital: Optional[float] = Field(None, description="初始现金金额")
    initial_position: Optional[int] = Field(None, description="初始股票仓位")

class UserIntent(BaseModel):
    """分类用户的意图并提取相应的参数。"""
    intent: Literal["analysis", "small_talk"] = Field(
        ...,
        description="将用户的意图分类为：'analysis' (如果他们想分析股票、获取财务数据或讨论特定公司) 或 'small_talk' (用于所有其他情况，如问候、一般性问题等)。"
    )
    parameters: Optional[AnalysisParameters] = Field(
        None,
        description="当且仅当意图是 'analysis' 时，在此处提取/推断参数。"
    )
    small_talk_response: Optional[str] = Field(
        None,
        description="当且仅当意图是 'small_talk' 时，在此处生成一个简短、友好的对话回应。"
    )

# --- 辅助函数：用提取的参数和默认值更新 data 字典 ---
def _update_data_with_defaults(data: dict, params: AnalysisParameters):
    """
    用新提取的参数更新 data 字典,
    并为缺失的日期和投资组合应用默认值。
    """
    updated_data = data.copy()

    # 1. 更新提取到的值
    if params.ticker:
        updated_data['ticker'] = params.ticker
        # 如果提供了 ticker, 公司名称的优先级就降低
        if 'company_name' in updated_data:
            updated_data.pop('company_name')

    elif params.company_name and not updated_data.get('ticker'):
        # 仅在没有 ticker 时才更新 company_name
        updated_data['company_name'] = params.company_name

    portfolio = updated_data.get('portfolio', {})
    if params.initial_capital:
        portfolio['cash'] = params.initial_capital
    if params.initial_position:
        portfolio['stock'] = params.initial_position
    updated_data['portfolio'] = portfolio

    # 2. 应用默认值 (逻辑来自您原来的 argparse)
    if 'cash' not in updated_data['portfolio']:
        updated_data['portfolio']['cash'] = 100000.0  # 默认
    if 'stock' not in updated_data['portfolio']:
        updated_data['portfolio']['stock'] = 0  # 默认

    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)

    end_date_str = params.end_date or updated_data.get('end_date')
    if end_date_str:
        end_date_obj = min(datetime.strptime(end_date_str, '%Y-%m-%d'), yesterday)
    else:
        end_date_obj = yesterday
    updated_data['end_date'] = end_date_obj.strftime('%Y-%m-%d')

    start_date_str = params.start_date or updated_data.get('start_date')
    if start_date_str:
        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
    else:
        start_date_obj = end_date_obj - timedelta(days=365)  # 默认1年
    updated_data['start_date'] = start_date_obj.strftime('%Y-%m-%d')

    if 'num_of_news' not in updated_data:
        updated_data['num_of_news'] = 20  # 默认

    return updated_data

# --- 意图识别 Agent ---
# --- 意图识别 Agent ---
def intent_recognition_agent(state: AgentState):
    """
    分析对话历史, 提取参数, 并决定是提问还是继续。
    """
    print("--- 节点: Intent Recognition ---")
    messages = state['messages']
    data = state['data'].copy()

    # 1. 初始化 LLM (请替换为您的模型名称)
    try:
        llm = create_model(
            provider='BYTEDANCE',
            name='deepseek-v3-1-250821',  # 确保这个模型在你的 config.py 中
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(UserIntent)
    except Exception as e:
        print(f"Error: 无法初始化 ChatByteDance: {e}")
        ai_msg = AIMessage(content="抱歉, 我的LLM未正确配置, 无法处理您的请求。")
        return {"messages": messages + [ai_msg]}
    # 2. 格式化对话历史
    conversation_str = "\n".join(
        [f"{m.type}: {m.content}" for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    )
    # 3. 创建提示词 (修复了之前的 Bug)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    你是一个复杂的金融分析系统的“前端大脑”。你的工作是理解用户的最新消息。
    你必须将用户的意图分为两类：
    1.  **analysis**: 用户想要分析股票、获取财务数据或讨论特定公司。
    2.  **small_talk**: 用户在闲聊、问候或询问与特定股票分析无关的一般性问题 (例如 "你好吗?", "什么是AI?,"斐波那契数列是什么？")。

    **关键指令:**
    -   如果意图是 **analysis**:
        -   你**必须**使用你的内部知识库，从公司名称**推断**出股票**ticker**。
        -   示例: '大智慧' -> '601519', '贵州茅台' -> '600519'。
        -   如果用户直接提供了 ticker，请使用它。
        -   提取任何其他参数 (日期、资金等)。
        -   **不要**生成回应 (response) 字段。
    -   如果意图是 **small_talk**:
        -   你**必须**生成一个友好、对话式的回应，。
        -   **不要**提取任何分析参数。

    这是我们之前对话中已收集到的数据:
    {current_data}

    请分析下面对话历史中的最后一条人类消息：
    """),
        ("human", "{conversation_history}")
    ])

    # 4. 运行 LLM 链
    chain = prompt | structured_llm
    try:
        intent_result = chain.invoke({
            "conversation_history": conversation_str,
            "current_data": data  # <--- 正确传递 data
        })
        print(f"LLM 意图分析结果: {intent_result}")

    except Exception as e:
        print(f"Error: LLM 意图提取失败: {e}")
        ai_msg = AIMessage(content="抱歉, 我在理解您的请求时遇到了点麻烦。您能换个方式再说一遍吗？")
        # 设置路由标志以等待用户
        data['__intent_route'] = "wait_for_user_input"
        return {"messages": messages + [ai_msg], "data": data}

    # 5. 处理结果并更新状态
    new_messages = messages

    if intent_result.intent == "analysis":
        if intent_result.parameters:
            # 使用您的辅助函数更新 data
            data = _update_data_with_defaults(data, intent_result.parameters)

            # 检查 LLM 是否成功推断出 Ticker
            if data.get('ticker'):
                ai_response = (
                    f"收到！已为您识别到股票 {data.get('company_name', '')} ({data['ticker']})。\n"
                    f"分析时段: {data['start_date']} 到 {data['end_date']}\n"
                    f"分析即将开始..."
                )
                new_messages = messages + [AIMessage(content=ai_response)]
                data['__intent_route'] = "start_analysis"  # 路由标志
            else:
                # LLM 认为意图是分析, 但无法推断 Ticker
                ai_response = "我很高兴为您运行分析。请问您对哪只股票感兴趣 (例如: 'AAPL' 或 '大智慧')？"
                new_messages = messages + [AIMessage(content=ai_response)]
                data['__intent_route'] = "wait_for_user_input"
        else:
            # 意图是分析, 但没有参数... 提问
            ai_response = "我很高兴为您运行分析。请问您对哪只股票感兴趣 (例如: 'AAPL' 或 '大智慧')？"
            new_messages = messages + [AIMessage(content=ai_response)]
            data['__intent_route'] = "wait_for_user_input"

    elif intent_result.intent == "small_talk":
        ai_response = intent_result.small_talk_response or "您好！有什么可以帮您的吗？"
        new_messages = messages + [AIMessage(content=ai_response)]
        data['__intent_route'] = "chit_chat"  # 路由标志

    return {
        "data": data,
        "messages": new_messages
    }


# --- 新增 Small Talk Agent ---
def small_talk_agent(state: AgentState):
    """
    处理一般性对话。
    """
    print("--- 节点: Small Talk ---")
    messages = state['messages']

    # 注意：在我们的新设计中, intent_recognition_agent 已经生成了回复
    # 并将其添加到了 messages 列表中。
    # 所以这个节点实际上只需要把流程导向 END 即可。
    # 如果 intent_agent 没有生成回复, 您可以在这里生成：

    # (如果 intent_agent 不负责生成回复, 则取消注释以下代码)
    # user_input = messages[-1].content
    # try:
    #     llm = ChatByteDance(model="your-byd-model-name", temperature=0.7)
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "你是一个专业的金融分析助手。用户正在和你闲聊。请友好地回复。"),
    #         ("human", "{question}")
    #     ])
    #     chain = prompt | llm
    #     response = chain.invoke({"question": user_input})
    #     ai_msg = AIMessage(content=response.content)
    #     return {"messages": messages + [ai_msg]}
    # except Exception as e:
    #     print(f"Error in small_talk_agent: {e}")
    #     return {"messages": messages + [AIMessage(content="我好像走神了，您能再说一遍吗？")]}

    # 鉴于 intent_agent 已经生成了回复, 我们什么都不用做
    return {}


# --- 新的路由函数 ---
# --- 修改后的路由函数 ---
def route_intent(state: AgentState):
    """
    根据 intent_recognition_agent 设置的标志来决定路由。
    """
    print("--- 路由: Routing from Intent ---")

    # 从 data 中读取路由标志
    route = state['data'].get('__intent_route')

    if route == "start_analysis":
        print("--- 决策: 意图是分析, Ticker 已找到。开始分析工作流。 ---")
        return "start_analysis"
    elif route == "chit_chat":
        print("--- 决策: 意图是闲聊。路由到 Small Talk 节点。 ---")
        return "chit_chat"
    else:  # "wait_for_user_input" 或 None
        print("--- 决策: 意图是分析, 但 Ticker 缺失。等待用户输入。 ---")
        return "wait_for_user_input"