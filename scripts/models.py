import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class User_feedback(BaseModel):
    id: str
    class_: str
    score: Optional[float] = None
    score_normalized: Optional[float] = None

class Extra_results(BaseModel):
    id: str

class DocumentRankedByModel(BaseModel):
    id: str
    score: float

class DocumentRankedManually(BaseModel):
    id: str
    score: int

class PytrecManualRankedQueries(BaseModel):
    id: str
    documents: List[DocumentRankedManually]

class PytrecModelRankedQueries(BaseModel):
    id: str
    documents: List[DocumentRankedByModel]

class PytrecEvaluation(BaseModel):
    manual_queries: List[PytrecManualRankedQueries]
    model_ranked_documents: List[PytrecModelRankedQueries]
    metrics: List[str]

class UlyssesManualQuery(BaseModel):
    id: str
    text: Optional[str] = None
    user_feedback: List[User_feedback]
    extra_results: Optional[List[Extra_results]] = None
    date_created: Optional[datetime.datetime] = None
    num_doc_feedback: Optional[int] = None
    extra_results_size: Optional[int] = None

class DocumentRankedByModelForUlysses(BaseModel):
    id: str
    score: float

class PytrecJustModelEvaluation(BaseModel):
    model_results: List[PytrecModelRankedQueries]
    metrics: List[str]
    path_to_dataset: str

class PytrecEvaluateByQueries(BaseModel):
    path:str
    index_name: str
    metrics: List[str]