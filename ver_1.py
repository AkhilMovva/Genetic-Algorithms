import numpy as np
import math


pop_size=1000
gen_size=200
outer_size=10

x1=np.random.uniform(-12,-6)
x2=np.random.uniform(12,6)

def cost_func(x1,x2): #cost caluclation funtion
    fy = 21.5 + (x1*math.sin(4*math.pi*x1)) + (x2*math.sin(20*math.pi*x2))
    return fy

def sel(x1_in, x2_in): #selection funtion
    
    temp1_idx=[]
    temp2_idx=[]
    x1_out = []
    x2_out = []
    for i in range(pop_size): # randoly choosing 2 children
        temp1_idx=np.random.choice(len(x1_in),pop_size,replace=False)
        temp2_idx=np.random.choice(len(x1_in),pop_size,replace=False)
    
    for t in temp1_idx:
        x1_out.append(x1_in[t])
    for t in temp2_idx:
        x2_out.append(x2_in[t])

            
    return x1_out, x2_out

def cross(x1_in, x2_in): #crossover funtion
    x1_out, x2_out = sel(x1_in, x2_in) 
    
    child1 = []
    child2 = []
    for j in range(pop_size//2):
        u=np.random.uniform(0,1)
  
        x1=x1_out[np.random.randint(0,pop_size)]
        x2=x2_out[np.random.randint(0,pop_size)]
        x1_1=x1_out[np.random.randint(0,pop_size)]
        x2_1=x2_out[np.random.randint(0,pop_size)]
        c11 = (x1 * u)+ (x1_1 * (1-u))  
        c12 = (x2 * u)+ (x2_1 * (1-u))
        c21 = (x1 * (1-u))+ (x1_1 * u)
        c22 = (x2 * (1-u))+ (x2_1 * u)
        child1.append(c11)
        child1.append(c21)
        child2.append(c12)
        child2.append(c22)
        
        
    return child1, child2
             

outer_y = np.zeros(gen_size)
outer_yavg=np.zeros(gen_size)
v1_ymax=[]
v1_x1max=[]
v1_x2max=[]

for out in range(outer_size):
    x1_in = [] 
    x2_in = []
    fy_max = []  
    fy_avg =[]
    child1_max = []
    child2_max = []

    for i in range(pop_size): 
        temp1 = np.random.uniform(-12, 12)
        temp2 = np.random.uniform(-6, 6)
        x1_in.append(temp1)
        x2_in.append(temp2)  

    for i in range(gen_size):
       
        child1, child2=cross(x1_in, x2_in)
    
        temp_ali1 =[]
        temp_ali2 =[]
        temp_fy =[]
        temp_fy_real=[]
   
        temp_ali1 = np.concatenate((x1_in, child1)) #concatenating parents with ofsprings
        temp_ali2 = np.concatenate((x2_in, child2))        
    
        for j in range(2*pop_size):
            temp_fy.append(cost_func(temp_ali1[j], temp_ali2[j])) #finding cost values
    
        temp_fy_real = np.vstack((temp_fy,temp_ali1, temp_ali2)) #stacking the list into 2d array
   
        temp_fy_real = temp_fy_real[:,temp_fy_real[0,:].argsort()] #sorting in ascending order
        temp_fy_real=np.flip(temp_fy_real,1) #fliping to get in desending order

        temp_fy_real=temp_fy_real[:,:(pop_size)] #Taking only first half of the array
  
        temp_fy_max, x1_in, x2_in = np.vsplit(temp_fy_real,(1*3)) #spling the 2d array into individual arrays

        temp_fy_max =np.reshape(temp_fy_max, (np.product(temp_fy_max.shape),)) #converting the array into list

        x1_in =np.reshape(x1_in, (np.product(x1_in.shape),))
        x2_in =np.reshape(x2_in, (np.product(x2_in.shape),))
        
        temp_fy =temp_fy_max
        
        fy_max.append(temp_fy_max[0])  #storing the max fitness values along with coresponding x1 and x2 values
        child1_max.append(x1_in[0])
        child2_max.append(x2_in[0])
        
        fy_avg.append(temp_fy.mean()) #caluclating mean and storing
        
    
    
    v1_ymax.append(np.amax(fy_max))  #finding the max fitness value
    max_index = np.where(fy_max == np.amax(fy_max)) # finding the max fitness value index
    idx=max_index[0][0]
    v1_x1max.append(child1_max[idx]) #storing the corespoding x1 and x2
    v1_x2max.append(child2_max[idx])

    
    
    outer_y = np.vstack((outer_y, fy_max)) # stacking the max fitness values vertically during each iteration
    outer_yavg=np.vstack((outer_yavg, fy_avg)) # stacking the average fitness values vertically during each iteration
    
        
outer_y = np.delete(outer_y, (0), axis=0)  #deleting the top row because it is initiated with zeros
outer_ymax_mean=outer_y.mean(axis=0)  #caluclating the mean of max values in vertically and storing
outer_ymax_std=outer_y.std(axis=0)    #caluclating the standard deviation of max values in vertically and storing

outer_yavg = np.delete(outer_yavg, (0), axis=0)  
outer_yavg_mean=outer_yavg.mean(axis=0) #caluclating the mean of avg values in vertically and storing
outer_yavg_std=outer_yavg.std(axis=0) #caluclating the standard deviation of avg values in vertically and storing
   

print("--------------Version 1-----------------")
print("-fy max-")
print(v1_ymax)
print("-x1-")
print(v1_x1max)
print("-x2-")
print(v1_x2max)






    



