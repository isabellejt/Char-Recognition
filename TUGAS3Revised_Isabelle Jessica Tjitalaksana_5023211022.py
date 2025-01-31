import numpy as np
import random

list_input = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
mse=0.5
count=0
learning_coeff=0.8

def preprocess(inp):
    desired_output=[]
    if inp == 'A':
        inp = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
            1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        desired_output=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        

    if inp == 'B':
        inp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        desired_output=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        

    if inp == 'C':
        inp = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        desired_output=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    if inp == 'D':
        inp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        desired_output=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        

    if inp == 'E':
        inp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        desired_output=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        

    if inp == 'F':
        inp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        desired_output=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        

    if inp == 'G':
        inp = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        desired_output=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        

    if inp == 'H':
        inp = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        desired_output=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    
    return inp, desired_output
        
neuron=[]
neuron.append(np.zeros(72))
neuron.append(np.zeros(36))
neuron.append(np.zeros(18))
neuron.append(np.zeros(8))
outlayer=neuron

weight=[]
weight.append(np.random.uniform(0,1,size=[144,72]))
weight.append(np.random.uniform(0,1,size=[72,36]))
weight.append(np.random.uniform(0,1,size=[36,18]))
weight.append(np.random.uniform(0,1,size=[18,8]))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        

def CalcLayer(i,origin,result):
    for j in range(len(result)):
        for k in range(len(origin)):
            neuron[i][j] += origin[k]*weight[i][k][j]
        outlayer[i][j] = sigmoid(neuron[i][j])
    return outlayer

def CalcMSE(output):
    mse=0
    for i in range(8):
        mse+=np.power((desired_output[i]-output[i]),2)
    return 0.5*mse

def gx(x):
    return x * (1 - x)

def UpdateWeight(num_layer,prev_layer,next_layer):
    for j in range(len(next_layer)):
        for k in range(len(prev_layer)):
            weight[num_layer][k][j]=weight[num_layer][k][j]+(learning_coeff*delta[num_layer][j]*prev_layer[k])
    return weight


for i in list_input:
    input,desired_output=preprocess(i)
    prev=input

    while mse>=0.0001:
        for num_layer in range(len(weight)):
            outlayer=CalcLayer(num_layer,input,neuron[num_layer])
            input=neuron[num_layer]

        #hitung delta
        delta=neuron
        for num_layer in range(3,-1,-1):
            if num_layer==3:
                for num_neuron in range(len(outlayer[3])):
                    delta[num_layer][num_neuron]=(desired_output[num_neuron]-outlayer[num_layer][num_neuron])*gx(outlayer[num_layer][num_neuron])
            else:
                for num_neuron in range(len(outlayer[num_layer])):
                    jmlh=0
                    for num_neuron_next in range(len(outlayer[num_layer+1])):
                        jmlh += delta[num_layer+1][num_neuron_next]*weight[num_layer][num_neuron][num_neuron_next]
                    delta[num_layer][num_neuron]=jmlh*gx(outlayer[num_layer][num_neuron])

        #hitung weights
        for num_layer in range(len(weight)):
            nextt=outlayer[num_layer]
            weight= UpdateWeight(num_layer,prev,nextt)
            prev=nextt

        mse = CalcMSE(outlayer[3])
        count+=1
        print(f'iter ke : {count}')
        print(mse)
        print(outlayer[3])

# LEARNING DONE
print("LEARNING DONE")

# RECALL
inp='A'
input,desired_output=preprocess(inp)
prev=input
for num_layer in range(len(weight)):
    outlayer=CalcLayer(num_layer,input,neuron[num_layer])
    input=neuron[num_layer]
print(outlayer[3])
