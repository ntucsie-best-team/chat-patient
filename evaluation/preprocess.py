import re
import pandas as pd
from typing import List, Dict

def preprocess_mts_dialog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the DataFrame of MTS-Dialog to a DataFrame of (context, question, answer) triplets and an ID field.
    """
    def dialogue_to_qa(dialogue: str) -> List[Dict[str, str]]:
        """
        Converts a dialogue string to a list of question-answer pairs.
        Return format: [{"doctor": "What is your name?", "patient": "My name is John."}, ...]
        """
        q_sents = re.findall(r"Doctor: (.*?)\s*(?=(?:Patient:)|\Z)", dialogue, re.DOTALL)
        a_sents = re.findall(r"Patient: (.*?)\s*(?=(?:Doctor:)|\Z)", dialogue, re.DOTALL)
        if len(q_sents) != len(a_sents):
            q_sents = q_sents[:len(a_sents)]
            a_sents = a_sents[:len(q_sents)]
        qa_pairs = [{"Q": q, "A": a} for q, a in zip(q_sents, a_sents)]
        return qa_pairs
    
    qa_pairs_l = df.dialogue.apply(dialogue_to_qa).tolist()
    id_l = []
    context_l = []
    question_l = []
    answer_l = []
    for i, qa_pairs in enumerate(qa_pairs_l):
        id_ = df.index[i]
        context = df.iloc[i].section_text
        for qa_pair in qa_pairs:
            question = qa_pair["Q"]
            answer = qa_pair["A"]
            question_l.append(question)
            answer_l.append(answer)
        id_l += [id_] * len(qa_pairs)
        context_l += [context] * len(qa_pairs)
    return pd.DataFrame({"ID": id_l, "context": context_l, "question": question_l, "answer": answer_l})

if __name__ == "__main__":
    name = "MTS-Dialog-TrainingSet"
    df = pd.read_csv(f"../data/MTS-Dialog/Main-Dataset/{name}.csv", index_col="ID")
    df = preprocess_mts_dialog(df)
    print(df.head())
