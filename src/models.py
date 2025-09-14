import openai
import tiktoken  
import torch,time
from typing import List
from transformers import T5Tokenizer, T5ForConditionalGeneration

class FlanT5():
    def __init__(
        self,
        model:str,
    ) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, device_map={"": torch.device("cuda:0")})

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
    
    def split_into_chunks(self,passage, chunk_size=64, max_chunks=2):
        tokens = self.tokenizer.tokenize(passage)
        chunks = [self.tokenizer.convert_tokens_to_string(tokens[i:i+chunk_size]) 
                    for i in range(0, min(len(tokens), chunk_size * max_chunks), chunk_size)]
        return chunks
    
    def generate(
        self,
        query: str,
        passages: List[str],
        max_length: int = 4096,
        temperature: float = 1.0
    ):
        query = self.truncate(query, 32)
        passages = [self.truncate(p, 128) for p in passages]
        n = len(passages)
        if n == 0:
            return 0, []
        label_ch=['A','B','C']
        passages = "\n\n".join([f'Passage {label_ch[i]}: "{passages[i]}"' for i in range(len(passages))])
        input_text =  f'Given a query "{query}", which of the following passages is the most relevant to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage:'
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        tokens = inputs.input_ids.shape[1] 
        label_texts = [f"<pad> Passage {label_ch[i]}" for i in range(n)]
        label_ids = self.tokenizer.batch_encode_plus(
            label_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        ).input_ids[:, -1].to(self.device) 
        decoder_start = self.tokenizer.encode(
            "<pad> Passage",
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=decoder_start
            )
            last_logits = outputs.logits[0, -1] 
        logits_list = [last_logits[idx].item() / temperature for idx in label_ids]
        return tokens, logits_list

class GPT:
    def __init__(self,
                 model:str="gpt-5",
                 max_retry:int=3,
                 timeout:int=30):
        self.model = model
        self.max_retry = max_retry
        self.timeout = timeout
        openai.api_key = "<YOUR_OPENAI_API_KEY>"
        self.encoding = tiktoken.encoding_for_model(model)
        self.label_all = "ABCDE" 
    
    def _trunc(self, text:str, max_tok:int)->str:
        toks = self.encoding.encode(text)
        return self.encoding.decode(toks[:max_tok])
    
    def generate(self,
                 query:str,
                 passages:List[str],
                 temperature:float=0.0):
        n = len(passages)
        if n == 0:
            return 0, []
        label_ch = list(self.label_all[:n])  
        system_prompt = (
            "You are a helpful passage-reranking assistant.\n"
            "Your task: select the SINGLE passage that is MOST relevant to the query.\n"
            f"You will receive ONE query and {n} candidate passages. "
            f"Each passage has a letter label {', '.join(label_ch)}.\n"
            "Return ONLY the letter label of the best passage — no other text or punctuation."
        )
        passage_block = "\n\n".join(
            f"[{label_ch[i]}] {passages[i]}" for i in range(n)
        )
        user_prompt = (
            f"Query:\n{query}\n\n"
            f"Candidate Passages:\n{passage_block}\n\n"
            f"Valid options: {', '.join(label_ch)}\n\n"
            "Answer (only one letter):"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
        tokens_in = len(self.encoding.encode(user_prompt))
        for attempt in range(self.max_retry):
            try:
                resp = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,             
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=10,  
                    timeout=self.timeout,
                )
                break
            except Exception:
                if attempt == self.max_retry - 1:
                    raise
                time.sleep(2 ** attempt)
        choice = resp.choices[0]
        tok_data = choice.logprobs.content[0]
        lp_dict = {item.token: item.logprob for item in tok_data.top_logprobs}
        if temperature == 0:
            logits_list = [lp_dict.get(ch, float("-inf")) for ch in label_ch]
        else:
            logits_list = [
                lp_dict.get(ch, float("-inf")) / temperature
                for ch in label_ch
            ]
        return tokens_in, logits_list