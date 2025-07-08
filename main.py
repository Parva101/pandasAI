import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.core.query_pipeline import QueryPipeline, Link, InputComponent
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
import pandas as pd

df = pd.read_csv("CHIZI_chatbot.csv")

class PandasWorkflow(Workflow):
    @step
    async def query_df(self, ev: StartEvent) -> StopEvent:
        instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)
        pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
        response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)
        pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
        response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
        llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        query_engine = PandasQueryEngine(df=df, 
                                         instruction_str=instruction_str,
                                         instruction_parser=PandasInstructionParser(df),
                                         pandas_prompt=pandas_prompt,
                                         llm=llm,
                                         synthesize_response=True,
                                         response_synthesis_prompt=response_synthesis_prompt,
                                         verbose=False)
        response = query_engine.query(ev.query)

        return StopEvent(result=str(response.response))

w = PandasWorkflow(timeout=30, verbose=True)
import asyncio
async def main():
    result = await w.run(query="summary for 26th june 2025?")
    print(result)

asyncio.run(main())