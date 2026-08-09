from typing import Any

from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_ollama import ChatOllama

from agents.analysis_tools import ANALYSIS_TOOLS
from agents.prompts import DATA_ANALYST_SYSTEM_PROMPT


def create_data_analyst_agent(
    model_name: str = "llama3.2:3b",
) -> AgentExecutor:
    """
    Create a LangChain 0.3-compatible tool-calling agent.

    Flow:
        user input
        -> ChatOllama
        -> tool selection
        -> Python analysis tool
        -> final explanation
    """

    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )

    # LangChain 0.3 tool-calling agents require a prompt
    # containing the agent_scratchpad placeholder.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                DATA_ANALYST_SYSTEM_PROMPT,
            ),
            MessagesPlaceholder(
                variable_name="chat_history",
                optional=True,
            ),
            (
                "human",
                "{input}",
            ),
            MessagesPlaceholder(
                variable_name="agent_scratchpad",
            ),
        ]
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=ANALYSIS_TOOLS,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=ANALYSIS_TOOLS,

        # Return readable tool errors instead of crashing
        # immediately when the model produces an invalid call.
        handle_parsing_errors=True,

        # Prevent an accidental endless tool loop.
        max_iterations=10,

        # Useful during development. Change to False later
        # if you do not want agent steps printed in the console.
        verbose=True,
    )


def ask_data_analyst(
    agent: AgentExecutor,
    question: str,
    chat_history: Any = None,
) -> str:
    """
    Send a question to the data analyst agent.

    Args:
        agent:
            AgentExecutor returned by
            create_data_analyst_agent().

        question:
            User's natural-language dataset question.

        chat_history:
            Optional LangChain message history.

    Returns:
        Final natural-language answer.
    """

    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    if not question.strip():
        raise ValueError(
            "question cannot be empty."
        )

    if chat_history is None:
        chat_history = []

    result = agent.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )

    return str(
        result["output"]
    )