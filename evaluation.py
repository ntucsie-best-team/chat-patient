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
        device: torch.device
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    # TODO: evaluate on a full dataset of (context, question, answer) triplets
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input format: a dataframe with the index field "ID" and the fields "context": str, "qa_pair": Dict[str, str] (keys: 'Q', 'A')
        Output format: the original dataframe with 3 additional fields "raw_score": Dict[str, float], "score": float, and "pred": str (one of "entailment", "neutral", "contradiction")
        """
        raise NotImplementedError
    
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
