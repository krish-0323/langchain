from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a tweet about the following {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write a linkedIn post about the following {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': prompt1 | model | parser,
    'linkedIn': prompt2 | model | parser,
})

result = chain.invoke({'topic': 'AI'})

print(result)
print()
print()
print(result['tweet'])
print()
print()
print(result['linkedIn'])