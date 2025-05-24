import streamlit as st
import time
from typing import Any, Mapping, List, Optional, Dict, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Extra, model_validator
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult, HumanMessage, SystemMessage
from vertexai.preview.language_models import TextGenerationResponse, ChatSession
from google.cloud import bigquery
from sqlalchemy import create_engine, MetaData
import pandas as pd
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Replace with your project ID and location
PROJECT_ID = "your-project-id"
LOCATION = "us-central1"

# Initialize VertexAI LLM
import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)

def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

class _VertexCommon(BaseModel):
    client: Any = None
    model_name: str = "text-bison@001"
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 200

    @property
    def _default_params(self) -> Mapping[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens
        }

    def _predict(self, prompt: str, stop: Optional[List[str]]) -> str:
        res = self.client.predict(prompt, **self._default_params)
        return self._enforce_stop_words(res.text, stop)

    def _enforce_stop_words(self, text: str, stop: Optional[List[str]]) -> str:
        if stop:
            return enforce_stop_tokens(text, stop)
        return text

    @property
    def _llm_type(self) -> str:
        return "vertex_ai"

class VertexLLM(_VertexCommon, LLM):
    model_name: str = "text-bison@001"


    @classmethod
    @model_validator(mode="before")
    def validate_model_name(cls, values):
        if not values.get("model_name"):
            raise ValueError("model_name is required")
        return values

    @classmethod
    @model_validator(mode="before")
    def validate_everything(cls, values):
        cls.validate_model_name(values)
        try:
            from vertexai.preview.language_models import TextGenerationModel
        except ImportError:
            raise ValueError("Could not import Vertex AI LLM python package.")

        try:
            values["client"] = TextGenerationModel.from_pretrained(values["model_name"])
        except AttributeError:
            raise ValueError("Could not set Vertex Text Model client.")

        return values

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._predict(prompt, stop)

@dataclass
class _MessagePair:
    question: HumanMessage
    answer: AIMessage

@dataclass
class _ChatHistory:
    history: List[_MessagePair] = field(default_factory=list)
    system_message: Optional[SystemMessage] = None

def _parse_chat_history(history: List[BaseMessage]) -> _ChatHistory:
    if not history:
        return _ChatHistory()
    first_message = history[0]
    system_message = first_message if isinstance(first_message, SystemMessage) else None
    chat_history = _ChatHistory(system_message=system_message)
    messages_left = history[1:] if system_message else history
    for question, answer in zip(messages_left[::2], messages_left[1::2]):
        if not isinstance(question, HumanMessage) or not isinstance(answer, AIMessage):
            raise ValueError("A human message should follow a bot one.")
        chat_history.history.append(_MessagePair(question=question, answer=answer))
    return chat_history

class VertexEmbeddings(Embeddings, BaseModel):
    model_name: str = "textembedding-gecko@001"
    model: Any
    requests_per_minute: int = 15

    @classmethod
    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from vertexai.preview.language_models import TextEmbeddingModel
        except ImportError:
            raise ValueError("Could not import Vertex AI LLM python package.")

        try:
            values["model"] = TextEmbeddingModel.from_pretrained(values["model_name"])
        except AttributeError:
            raise ValueError("Could not set Vertex Text Model client.")

        return values

    class Config:
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.model = self.model.from_pretrained(self.model_name)
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            head, docs = docs[:2], docs[2:]
            chunk = self.model.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]

    def embed_query(self, text: str) -> List[float]:
        single_result = self.embed_documents([text])
        return single_result[0]

llm = VertexLLM(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# BigQuery connection details
client = bigquery.Client(project=PROJECT_ID)
dataset_id = "iowa_liquor_sales"
table_name = "sales"

# Create engine for SQLAlchemy
table_uri = f"bigquery://{PROJECT_ID}/{dataset_id}"
engine = create_engine(table_uri)

query=f"""SELECT * FROM `{PROJECT_ID}.{dataset_id}.{table_name}`"""
engine.execute(query).first()

def bq_qna(question):
    """
    Executes BigQuery query using LangChain and SQLAlchemy
    """
    # Create SQL Database instance from BigQuery engine
    db = SQLDatabase(engine=engine, metadata=MetaData(), include_tables=[table_name])

    # Create SQL DB Chain with LLM and SQLDatabase instance
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)

    # Define prompt for BigQuery SQL
    _googlesql_prompt = """
    You are a BigQuery SQL expert. Given an input question, first create a syntactically correct BigQuery query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per BigQuery SQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Use the following format:
    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"
    Only use the following tables:
    {table_info}

    Question: {input}
    """

    BigQuerySQL_PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template=_googlesql_prompt,
    )

    # Pass question to the prompt template
    final_prompt = BigQuerySQL_PROMPT.format(input=question, table_info=table_name, top_k=10000)

    # Pass final prompt to SQL Chain
    output = db_chain(final_prompt)

    return output["result"]

def print_table(query):
    """
    Executes a BigQuery query and displays the results as a table in Streamlit
    """
    with engine.connect() as connection:
        result = connection.execute(query)
        df = result.fetchall()
        # Convert results to DataFrame
        df = pd.DataFrame(df, columns=result.keys())
    st.dataframe(df)


st.title("BigQuery Q&A with LangChain")

user_question = st.text_input("Enter your question about the sales data:")

if user_question:
    print_table_option = st.checkbox("Print the table results?")
    result = bq_qna(user_question)

    st.write("**Answer:**")
    st.write(result["Answer"])

    if print_table_option:
        print_table(result["SQLQuery"])
