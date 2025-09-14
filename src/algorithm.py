import copy
import time
import torch
import random
import math
import trueskill
from tqdm import tqdm
from scipy import stats
from models import FlanT5,GPT

inf=0
depth=0
ranker = ""
num_tokens = 0

def format_result(result):
    run = {}
    for qid, docids in result.items():
        run[qid] = {
            docid: int(len(docids) - i)
            for i, docid in enumerate(docids)
        }
    return run

def score(rating):
    return rating.mu

def update_rating(rating1, rating2,env,p):
    r1_win, _      = env.rate_1vs1(rating1, rating2)
    _,      r1_loss = env.rate_1vs1(rating2, rating1)
    mu_w, mu_l   = r1_win.mu,   r1_loss.mu
    var_w, var_l = r1_win.sigma**2, r1_loss.sigma**2
    mu_new = p * mu_w + (1 - p) * mu_l
    sec    = p * (var_w + mu_w**2) + (1 - p) * (var_l + mu_l**2)
    sigma_new = math.sqrt(max(1e-9, sec - mu_new**2))
    return env.create_rating(mu=mu_new, sigma=sigma_new)

def soft_aggregate(env, clones):
    tau_sum = sum(1/(r.sigma**2) for r in clones)
    mu_agg  = sum(r.mu/(r.sigma**2) for r in clones) / tau_sum
    sigma_agg = (1/(tau_sum/len(clones)))**0.5
    return env.create_rating(mu=mu_agg, sigma=sigma_agg)

def logits_to_prob(logit_A,logit_B):
    T=34
    diff = (logit_A - logit_B) / T
    tensor_diff = torch.tensor(diff, device='cpu')
    return torch.sigmoid(tensor_diff).item()

def logits_to_prob_GPT(logit_A: float, logit_B: float):
    temperature: float = 720
    if math.isinf(logit_A) and math.isinf(logit_B):
        return 0.5
    diff = (logit_A - logit_B) / max(temperature, 1e-8)
    return 1.0 / (1.0 + math.exp(-diff))

async def realm(data,llm,order):
    global ranker,inf, depth
    if "gpt" in llm:
        ranker= GPT(model=llm)
    else:
        ranker = FlanT5(model=llm)
    inf=depth=number=ttime=0
    res={}
    env = trueskill.TrueSkill()
    for entry in tqdm(data, desc="Processing queries", unit="query"):
        number+=1
        query = entry['query']
        docs = entry['hits']
        if len(docs)==0:
            continue
        id=str(docs[0]['qid'])
        result=[]
        if order=="random":
            random.shuffle(docs)
        elif order=="inverse":
            for i in range(len(docs)//2):
                docs[i],docs[len(docs)-i-1]=docs[len(docs)-i-1],docs[i]
        for i in range(len(docs)):
            docs[i]['elo']=env.create_rating(mu=25-i*0.001)
        for i in range(len(docs)):
            result.append(i)
        
        async def partition(arr, low, high):
            global depth
            depth+=1
            x=[]
            for j in range(low, high+1):
                x.append(docs[arr[j]]['docid'])
            left=low
            right=high
            mid=left
            sig=docs[arr[left]]['elo'].sigma
            for j in range(left, right+1):
                if docs[arr[j]]['elo'].sigma<sig:
                    mid=j
                    sig=docs[arr[j]]['elo'].sigma
            pivot = arr[mid]
            arr[mid], arr[high] = arr[high], arr[mid]
            clones=[]
            j = low
            while j < high:
                global inf
                inf+=1
                candidate=[docs[arr[j]]['content']]
                while j+1 < high:
                    j+=1
                    candidate.append(docs[arr[j]]['content'])
                    break
                candidate.append(docs[pivot]['content'])
                lll=len(candidate)
                tokens,r= ranker.generate(query, candidate)
                global num_tokens
                num_tokens +=tokens
                for i in range(len(candidate)-1):
                    if "gpt" in llm:
                        p=logits_to_prob_GPT(r[i],r[lll-1])
                    else:
                        p=logits_to_prob(r[i],r[lll-1])
                    pos=j-(lll-2)+i
                    docs[arr[pos]]['elo']=update_rating(docs[arr[pos]]['elo'],copy.deepcopy(docs[pivot]['elo']),env,p)
                    clones.append(update_rating(copy.deepcopy(docs[pivot]['elo']),docs[arr[pos]]['elo'],env,1-p))
                j+=1
            if len(clones)>0:
                docs[pivot]['elo']=soft_aggregate(env,clones)
            arr[low:high+1] = sorted(arr[low:high+1], key=lambda x: score(docs[x]['elo']), reverse=True)
            for j in range(low,high+1):
                if arr[j]==pivot:
                    i=j
                    break
            y=[]
            for j in range(low, high+1):
                y.append(docs[arr[j]]['docid'])
            tau, p = stats.kendalltau(x, y, nan_policy="omit")
            return tau>0.95,(i*2+((low+high))//2)//3

        async def quick_sort(arr, low, high):
            if low < high:
                if high<10:
                    return
                stop,pi = await partition(arr, low, high)
                if stop==1:
                    return
                await quick_sort(arr, low, pi-1)
                if pi+1 <10:
                    await quick_sort(arr, pi + 1, high)
        
        start = time.perf_counter()
        await quick_sort(result, 0, len(result) - 1)
        end = time.perf_counter()
        ttime+=end-start
        res[id]=[]
        for i in range(len(result)):
            res[id].append(docs[result[i]]['docid'])

    print("\n")
    print("Inference Count:", inf/number)
    print("Tokens in Prompt:", num_tokens/number)
    print("Latency (s):", ttime/number)
    print("Depth:", depth/number)
    return res