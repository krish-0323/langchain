from langchain_core import chat_history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

chat_template = ChatPromptTemplate([
    ('system', 'You are an helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

prompt = chat_template.invoke({'chat_history': chat_history, 'query':'Where is my refund'})

print(prompt)