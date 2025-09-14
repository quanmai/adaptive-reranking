import json
import asyncio
import argparse
import pytrec_eval
from algorithm import realm, format_result
from quantum_inspired_adr import quantum_inspired_adr

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate REALM and QI-ADR algorithms."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="dl19",
        help="Dataset name (used to locate ../data/retrieve_results_{dataset}.json and qrels_{dataset}.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="google/flan-t5-large",
        choices=["google/flan-t5-large","google/flan-t5-xl","google/flan-t5-xxl","gpt-5"],
        help="Which LLM to use."
    )
    parser.add_argument(
        "--order", "-o",
        type=str,
        default="bm25",
        choices=["bm25","random","inverse"],
        help="Initial order before reranking."
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="realm",
        choices=["realm", "qi-adr"],
        help="Which algorithm to use: REALM or Quantum-Inspired ADR."
    )
    parser.add_argument(
        "--llm-budget",
        type=int,
        default=20,
        help="Maximum number of LLM evaluations for QI-ADR (ignored for REALM)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of documents to evaluate per iteration in QI-ADR."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top documents to return."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for state vector updates in QI-ADR."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model for QI-ADR."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset
    input_path='../data/retrieve_results_'+dataset+'.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qrels_path='../data/qrels_'+dataset+'.json'
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    for key in qrels:
        for doc in qrels[key]:
            qrels[key][doc]=int(qrels[key][doc])
    for i in range(len(data)):
        docs = data[i]['hits']
        if len(docs)==0:
            continue
        id=str(docs[0]['qid'])
        n_items = len(docs)
        for j in range(n_items):
            if docs[j]['docid'] in qrels[id]:
                data[i]['hits'][j]['qrels']=qrels[id][docs[j]['docid']]
            else:
                data[i]['hits'][j]['qrels']=0
    
    # Run the selected algorithm
    if args.algorithm == "realm":
        print(f"Running REALM algorithm with {args.model}")
        result = asyncio.run(realm(data, args.model, args.order))
    elif args.algorithm == "qi-adr":
        print(f"Running Quantum-Inspired ADR with {args.model}")
        print(f"LLM Budget: {args.llm_budget}, Batch Size: {args.batch_size}")
        result = asyncio.run(quantum_inspired_adr(
            data=data,
            llm=args.model,
            order=args.order,
            batch_size=args.batch_size,
            llm_budget=args.llm_budget,
            top_k=args.top_k,
            learning_rate=args.learning_rate,
            embedding_model_name=args.embedding_model
        ))
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    run = format_result(result)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    evaluation_result = evaluator.evaluate(run)
    avg_ndcg = sum(res['ndcg_cut_10'] for res in evaluation_result.values()) / len(evaluation_result)
    print(f"\n{args.algorithm.upper()} NDCG@10: {avg_ndcg:.4f}\n")

if __name__ == "__main__":
    main()