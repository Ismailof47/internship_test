from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Инструмент 1: Поиск в интернете
search = DuckDuckGoSearchAPIWrapper()
search_tool = Tool.from_function(
    name="web_search",
    func=search.run,
    description="Полезен для поиска актуальной информации в интернете"
)

# Инструмент 2: Калькулятор
def calculator(query: str) -> str:
    try:
        return str(eval(query))  # Для демонстрации (в продакшене используйте безопасные методы!)
    except:
        return "Ошибка вычисления"

calc_tool = Tool.from_function(
    name="calculator",
    func=calculator,
    description="Полезен для математических вычислений, например: '2+2' или 'цена * 0.2'"
)

!!!!!!

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Промпт для агента (шаблон рассуждений)
prompt = ChatPromptTemplate.from_template("""
Ты — умный помощник. Отвечай на вопросы, используя инструменты если нужно.

Доступные инструменты:
{tools}

Инструкции:
1. Думай шаг за шагом.
2. Если нужен инструмент — используй его.
3. Если инструмент не нужен — отвечай сразу.

Вопрос: {input}
""")

# Инициализация LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Собираем агента
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, calc_tool],
    prompt=prompt
)

# Обёртка для выполнения
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, calc_tool],
    verbose=True  # Показывать логи работы
)


from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    output: str | None

def agent_node(state: AgentState):
    result = agent_executor.invoke({"input": state["input"]})
    return {"output": result["output"]}

# Создаём граф
workflow = StateGraph(AgentState)
workflow.add_node("ask_agent", agent_node)
workflow.set_entry_point("ask_agent")
workflow.set_finish_point("ask_agent")

app = workflow.compile()

# Запускаем
app.invoke({"input": "Сколько будет 2+2?"})

Если граф завершился с ошибкой, состояние будет содержать поле "error".

Для сложных графов используйте verbose=True при компиляции:

python
Copy
app = workflow.compile(debug=True)  # Подробные логи
