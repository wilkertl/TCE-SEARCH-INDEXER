from models import *
import pytrec_eval
import pandas as pd
import ast
from pydantic import BaseModel, Field

def relevance_to_int(relevance: str) -> int: #r - relevant, pr - partially relevant, else is irrelevant
    if relevance == "r" or relevance == "pr":
        return 1
    else:
        return 0
    
def format_metric(metricValue: dict, metricResults: dict) -> None:
    
    for outerKey, innerDictionary in metricValue.items():
        for key, value in innerDictionary.items():
            if(key not in metricResults):
                metricResults[key] = {
                    "value": 0,
                    "qtdItems": 0
                }
            metricResults[key]["value"] += value
            metricResults[key]["qtdItems"] += 1
    
        
def mean_metric(metricResults: dict) -> dict:
    results = {}
    
    for key, value in metricResults.items():
        if(value["qtdItems"] > 0):
            results[key] = value["value"] / value["qtdItems"]
    
    return results 

def format_pytrec_evaluation(to_evaluate: PytrecEvaluation):
    
    qrel = {}
    run = {}
    metrics = to_evaluate.metrics

    manual_queries: List[PytrecManualRankedQueries] = to_evaluate.manual_queries
    model_ranked_documents: List[PytrecModelRankedQueries] = to_evaluate.model_ranked_documents 

    for manual_query in manual_queries:
        q = {}
        for document in manual_query.documents:
            q[document.id] = int(round(document.score))
        qrel[manual_query.id] = q

    for model_rank in model_ranked_documents:
        r = {}
        for document_ranked_by_model in model_rank.documents:
            r[document_ranked_by_model.id] = float(document_ranked_by_model.score)
        run[model_rank.id] = r

    return {
        'qrel': qrel,
        'run': run,
        'metrics': metrics
    }

def evaluate(qrel: List[PytrecModelRankedQueries], model_results: List[PytrecModelRankedQueries], metrics: List[str]):
    
    to_evaluate = PytrecEvaluation(
        manual_queries = qrel,
        model_ranked_documents = model_results,
        metrics = metrics
    )
    
    data_to_evaluate = format_pytrec_evaluation(to_evaluate=to_evaluate)
    metric_results = {}
    
    evaluator = pytrec_eval.RelevanceEvaluator(data_to_evaluate.get('qrel'), data_to_evaluate.get('metrics'))
    
    result = evaluator.evaluate(data_to_evaluate.get('run'))

    format_metric(metricValue=result, metricResults=metric_results)

    evaluation_result = {
        "mean_metric_results": mean_metric(metricResults=metric_results),
        "individual_response": result
    }

    return evaluation_result

def UlyssesQueryToPytrecQuery(data: UlyssesManualQuery):
    id = data.id
    user_feedback = data.user_feedback
    extra_results = data.extra_results

    documents = []

    for query in user_feedback:
        documents.append(
            DocumentRankedManually(
                id=query.id, score=relevance_to_int(query.class_)
            )
        )
    for query in extra_results:
        documents.append(
            DocumentRankedManually(id=query.id, score=1)
        )

    return PytrecManualRankedQueries(
        id=id,
        documents=documents
    )

def user_feedback_from_ulysses(data):
    list_of_feedback: List[User_feedback] = []

    should_append = False

    qtd = 0
    for document in data:
        if(document.get("class") == "pr"):
            continue
        
        if(qtd >= 10):
            break

        if(document.get("class") == "r"):
            should_append = True

        info = User_feedback(
                id = document.get("id"),
                class_ = document.get("class")
            )
        list_of_feedback.append(info)
        qtd += 1

    return list_of_feedback, should_append

def extra_results_from_ulysses(data):
    list_of_extra_results: List[Extra_results] = []

    # for document in data:
    #     info = Extra_results(id = document)
    #     list_of_extra_results.append(info)

    return list_of_extra_results


def preprocess_json_string(json_str: str) -> str:
    json_str = json_str.replace("'", '"')

    if json_str.endswith(","):
        json_str = json_str[:-1]
        
    return json_str

def not_null_column_csv(input:str):
    return ast.literal_eval(input)

def qrel_from_ulysses(path:str) -> List[PytrecManualRankedQueries]:
    try:
        df = pd.read_csv(path, usecols=['id', 'query', 'user_feedback', 'extra_results'], delimiter=',', index_col=False, encoding='utf-8')

    except:
        raise Exception('Documento não foi encontrado no path especificado')

    ulysses_queries: List[UlyssesManualQuery] = []

    for index, row in df.iterrows():     

        ex_str = preprocess_json_string(str(row["extra_results"]))
        u_feedback_str = preprocess_json_string(str(row["user_feedback"]))
        
        u_feedback = not_null_column_csv(u_feedback_str)
        ex_results = not_null_column_csv(ex_str)

        id_ulysses = str(row["query"])
        user_feedback_ulysses, should_append = user_feedback_from_ulysses(u_feedback)
        extra_results_ulysses = extra_results_from_ulysses(ex_results)

        if(not should_append):
            continue

        ulysses_queries.append(
            UlyssesManualQuery(
                id = id_ulysses, 
                user_feedback = user_feedback_ulysses,
                extra_results = extra_results_ulysses
            )
        )

    qrel: List[PytrecManualRankedQueries] = []
    for query in ulysses_queries:
        qrel.append(UlyssesQueryToPytrecQuery(query))

    return qrel

def read_without_nan(input):
    return None if pd.isna(input) else input
