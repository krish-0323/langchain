from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Give a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

gen_report_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = gen_report_chain | branch_chain

result = chain.invoke({'topic': 'Russia vs Ukraine'})

print(result)

