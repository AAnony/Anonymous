import openai, time
import pandas as pd
import csv, codecs
import json

# openai.api_base = "https://api.chatanywhere.cn/v1"
# openai.api_key = "sk-EaE4ZZsGm0ffofkn4oYz3z8iO8GhH21GlN24UwZLNTu02PH1"

# openai.api_base = "https://api.openai.com/v1/chat/completions"
# openai.api_key="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"
# proxy = {
# 'http': 'http://localhost:7890',
# 'https': 'http://localhost:7890'
# }

def generate_msg(que:str, template:str, ans:list):
    ret = '<Question>:{' + que + '}\n<template>:{' + template + "}\n"
    idx = 1
    for item in ans:
        ret += "<Answer" + str(idx) + ">:{\n" + item + "\n}\n"
        idx += 1
    return ret

def get_score(que:str, template:str, ans:list):
    msgs=[
        {"role": "system", "content": 'You are a code grader, and I will provide you with this format: <Question>:{}\n<template>:{}\n<Answer1>:{}\n<Answer2>:{}\n...<Answerx>:{}\n. <template> is the correct code. Based on the code problem described in the <Question> and <template>, you need to grade each <Answer>. You need to refer to <template>, and consider the correctness and readability of the code. The output should be the grades(from 1 to 10) for each answer separated by ,'},
        # {"role": "user", "content": '<Question>:{Count up from 1 to 500}\n<Answer1>:{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}\n<Answer2>:{"for i in range(1,501):\n    print(i)"}'},
        {"role": "user", "content": generate_msg("Count up from 1 to 500","for i in range(1,501):\n    print(i)",["1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14","for i in range(1,501):\n    print(i)"])},
        {"role": "assistant", "content": "1,10"},
        {"role": "user", "content": generate_msg(que=que, template = template, ans=ans)}
    ]
    # print(msgs)
    time.sleep(2)
    #gpt-3.5-turbo
    #gpt-4-0613
    chat_completion = openai.ChatCompletion.create(model="gpt-4-0613", messages=msgs,temperature=0.1)
    return chat_completion.choices[0].message.content

# ChatCompletion适用于生成对话和聊天场景的文本，
# Completion则适用于更为广泛的自然语言生成场景。
ques = "Write a Python program that prints the first 10 Fibonacci numbers"
template = "def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n - 1) + fibonacci(n - 2)"
ans = ["def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n - 1) + fibonacci(n - 2)","numbers = [0, 1]\n\nfor i in range(2, 11):\n    numbers.append(numbers[i-1] + numbers[i-2])\n\nprint(numbers)"]
# print(ques,template,ans)



pds = []


# with open(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\20k_onlypython_llama_Q.jsonl", 'r') as f:
#     pd_q = json.load(f)
# pd_q = json.dumps(pd_q)
# pd_q = pd.read_json(pd_q)

# with open(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\ori_llama.jsonl", 'r') as f:
#     pd_a1 = json.load(f)
# pd_a1 = json.dumps(pd_a1)
# pd_a1 = pd.read_json(pd_a1)

# with open(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\ori_alpaca.jsonl", 'r') as f:
#     pd_a2 = json.load(f)
# pd_a2 = json.dumps(pd_a2)
# pd_a2 = pd.read_json(pd_a2)

# with open(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\2k_onlypython_llama.jsonl", 'r') as f:
#     pd_a3 = json.load(f)
# pd_a3 = json.dumps(pd_a3)
# pd_a3 = pd.read_json(pd_a3)

# with open(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\20k_onlypython_llama_Sum.jsonl", 'r') as f:
#     pd_a4 = json.load(f)
# pd_a4 = json.dumps(pd_a4)
# pd_a4 = pd.read_json(pd_a4)

# with open(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\20k_onlypython_llama_C.jsonl", 'r') as f:
#     pd_a5 = json.load(f)
# pd_a5 = json.dumps(pd_a5)
# pd_a5 = pd.read_json(pd_a5)





pd_q = pd.read_json(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\20k_onlypython_llama_Q.jsonl", orient="records", lines= True)
# pds.append(pd_q)
pd_a1 = pd.read_json(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\ori_llama.jsonl", orient="records", lines= True)
pds.append(pd_a1)
pd_a2 = pd.read_json(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\ori_alpaca.jsonl", orient="records", lines= True)
pds.append(pd_a2)
pd_a3 = pd.read_json(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\2k_onlypython_llama.jsonl", orient="records", lines= True)
pds.append(pd_a3)
pd_a4 = pd.read_json(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\20k_onlypython_llama_Sum.jsonl", orient="records", lines= True)
pds.append(pd_a4)
pd_a5 = pd.read_json(r"F:\study\postgraduate\paper\SE\FLLLM\代码\gpt_pack_llama_20k_onlypython\20k_onlypython_llama_C.jsonl", orient="records", lines= True)
pds.append(pd_a5)

# exit()
scores = [0]*pds.__len__()
for i in range(0, len(pd_q)):
    ques = pd_q.loc[i].instruction
    extra = ""
    if pd_q.loc[i].input:
        extra = " Input:"+pd_q.loc[i].input
    ques = ques + extra
    template = pd_q.loc[i].output
    ans = []
    for item in pds:
        ans.append(item.loc[i].completion)
    score = []
    inp = ""
    # score = get_score(ques,template,ans).split(",")
    while inp != "Q":
        try:
            score = get_score(ques,template,ans).split(",")
            inp = "Q"
        except:
            inp = input("fail, enter Q to leave, or other to continue\n")
            score = []
            if inp == "Q":
                exit()
    print(score)
    for idx in range(0, len(scores)):
        scores[idx] += int(score[idx].strip())
    
print(scores)

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
        file_csv = codecs.open(file_name,'w+','utf-8')#追加
        writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(data)
        print("保存csv文件成功，处理结束")
