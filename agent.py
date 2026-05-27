from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama

# ✅ Use your existing LLM
llm = Ollama(model="llama3")

# ✅ TOOL 1: Summarizer
def summarize_tool(text):
    return f"Summary:\n{text[:300]}"

# ✅ TOOL 2: Quiz generator
def quiz_tool(text):
    return "Quiz:\n1. What is the main idea?\n2. Explain key concept."

# ✅ Tool definitions
tools = [
    Tool(
        name="Summarizer",
        func=summarize_tool,
        description="Use this when user asks to summarize content"
    ),
    Tool(
        name="QuizGenerator",
        func=quiz_tool,
        description="Use this when user asks to generate questions or quiz"
    )
]

# ✅ Create agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# ✅ Main function
def run_agent(query, context):
    full_input = f"User Query: {query}\nContext: {context}"
    return agent.run(full_input)