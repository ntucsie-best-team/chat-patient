from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any
from abc import ABC, abstractmethod

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FaithfulnessEvaluator(ABC):

    @abstractmethod
    def evaluate_qa(
        self,
        context: str,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate the faithfulness of the answer to the question given the context.
        Return format:
        {
            "raw": <raw object>, # e.g., {"entailment": 0.1, "contradiction": 0.9, "neutral": 0.0}
            "score": <score>, # e.g., 0.0
        }
        """
        raise NotImplementedError

class E2ENLIEvaluator(FaithfulnessEvaluator):
    label_names = ["entailment", "neutral", "contradiction"]
    
    def __init__(
        self,
        model_name: str,
        device: str
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    # TODO: evaluate on a full dataset of (context, question, answer) triplets
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input format: a dataframe with the fields "ID": int, "context": str, "question": str, "answer": str
        Output format: the original dataframe with 3 additional fields "raw_score": Dict[str, float], "score": float, and "pred": str (one of "entailment", "neutral", "contradiction")
        """
        pbar = tqdm(total=len(df))
        def _evaluate_qa(row):
            pbar.update(1)
            return self.evaluate_qa(row.context, row.question, row.answer)
        res = df.apply(_evaluate_qa, axis=1).tolist()
        pbar.close()

        fields = list(res[0].keys())
        for field in fields:
            df[field] = [r[field] for r in res]

        return df
    
    def evaluate_qa(
        self,
        context: str,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate the faithfulness of the answer to the question given the context.
        The (context, question, answer) triplet is converted to a (hypothesis, premise) pair, and an NLI model is used to evaluate the faithfulness.
        Return format:
        {
            "raw": {"entailment": <float>, "contradiction": <float>, "neutral": <float>}
            "score": <float>
        }
        """
        premise, hypothesis = self._nli_preprocess(context, question, answer)
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self.device)
        output = self.model(inputs.input_ids)
        preds = torch.softmax(output.logits[0], dim=-1).tolist()
        preds = {label_name: float(pred) for label_name, pred in zip(self.label_names, preds)}
        score = self.preds_to_score(preds)
        return {"raw_score": preds, "score": score, "pred": max(preds, key=preds.get)}
    
    @staticmethod
    def preds_to_score(
        preds: Dict[str, float]
    ) -> float:
        """
        Convert the raw output of the NLI model to a single score.
        """
        label2score = {
            "entailment": 1.0,
            "neutral": 0.5,
            "contradiction": 0.0
        }
        pred = max(preds, key=preds.get)
        return label2score[pred]

    @staticmethod
    def _nli_preprocess(
        context: str,
        question: str,
        answer: str
    ):
        return context, question + " " + answer

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tsv_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True, help="Path to save the evaluation results")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(f"Initializing evaluator with model {args.model_name}...")
    evaluator = E2ENLIEvaluator(args.model_name, args.device)
    df = pd.read_csv(args.tsv_path, sep="\t")
    print(f"Evaluating on {len(df)} examples...")
    edf = evaluator(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    edf.to_csv(args.out_dir / args.tsv_path.name, sep="\t", index=False)
