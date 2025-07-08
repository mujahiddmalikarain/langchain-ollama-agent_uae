from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from tools.weather_tool import get_weather
from tools.developer_tool import get_developer_info

def create_main_agent():
    llm = Ollama(model="llama3")

    tools = [
        Tool(
            name="WeatherTool",
            func=get_weather,
            description="Use this to get weather info for a city."
        ),
        # Tool(
        #     name="DeveloperTool",
        #     func=get_developer_info,
        #     description="Use this to get real estate developer info by name."
        # )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True  
    )

    return agent
