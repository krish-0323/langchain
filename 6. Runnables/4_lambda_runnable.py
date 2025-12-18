from importlib.abc import PathEntryFinder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from pydantic.type_adapter import R

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write an joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

joke_gen_chain = prompt | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

chain = joke_gen_chain | parallel_chain

result = chain.invoke({'topic': 'Cricket'})

print(result)