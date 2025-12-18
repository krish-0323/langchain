from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain me in simple terms, what/who is {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'hitman'})

print(prompt)



