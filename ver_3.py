import numpy as np
import math


pop_size=1000 # population size
gen_size=200 # number of generations

outer_size=10 # number of random seeds

x1=np.random.uniform(-12,-6)
x2=np.random.uniform(12,6)

def cost_func(x1,x2): #cost function
    fy = 21.5 + (x1*math.sin(4*math.pi*x1)) + (x2*math.sin(20*math.pi*x2))
    return fy

def sel_rd(x1_in, x2_in): #selection function by using runiform random distribution
    x1_out = x1_in
    x2_out = x2_in
    
    np.random.shuffle(x1_out)
    np.random.shuffle(x2_out)
    
    x1_out, x2_out = sel_pd(x1_out, x2_out)
    return x1_out, x2_out
    

def sel_pd(x1_in, x2_in): #selection function by using fitness percentage
    x1_out = x1_in
    x2_out = x2_in
    pd=[]
    cost=[]
    for i in range (pop_size):
        cost.append(cost_func(x1_out[i], x2_out[i]))
    
    pd=np.true_divide(cost,sum(cost))
    
    x1_out = np.random.choice(x1_in, pop_size,
              p=pd, replace = True)
    x2_out = np.random.choice(x2_in, pop_size,
              p=pd, replace = True)
            
            
    return x1_out, x2_out



def cross(x1_in, x2_in): #crossover function
    x1_out, x2_out = sel_pd(x1_in, x2_in) 
    x1_out, x2_out = sel_rd(x1_out, x2_out)

    temp_child1 = x1_out[:(pop_size//2)] #storing the first half of the list values
    temp_child2 = x2_out[:(pop_size//2)]
    
    child1 = temp_child1.tolist()
    child2 = temp_child2.tolist()
    
    for j in range(pop_size//4): # caluclating the second half by using crossover equations
        u=np.random.uniform(0,1) #crossover rate
  
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
    child1, child2 = mut(child1, child2)
    
    return child1, child2
        
def mut(x1_in, x2_in): #mutaion funtion
    x1_out= x1_in
    x2_out= x2_in 
    child1 = []
    child2 = []
    u_mut=0.1 #mutaion rate
    for j in range(pop_size):
        u1=np.random.uniform(0,1)
        u2=np.random.uniform(0,1)
        if (u1<u_mut) and (x1_out[j]+u1)<12 and (x1_out[j]+u1)>-12:
            child1.append((x1_out[j]+u1))      #comparing the mutaion  rate along with the x1 bound    
        else:   
            child1.append(x1_out[j])
            
        if (u2<u_mut) and (x2_out[j]+u2)<6 and (x2_out[j]+u2)>-6:
            child2.append((x2_out[j]+u2))      #comparing the mutaion  rate along with the x2 bound  
        else:      
            child2.append(x2_out[j])
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
   

print("--------------Version 3-----------------")
print("-fy max-")
print(v1_ymax)
print("-x1-")
print(v1_x1max)
print("-x2-")
print(v1_x2max)






    



