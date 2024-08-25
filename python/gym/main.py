import sysv_ipc as ipc
import gymnasium as gym

import numpy as np

import os    



def readInterface():
    
    
    output = []
    
    with open("../../fifo","r") as fifo:
        line = fifo.readline()
        
        values = line.split(";")
        
        for value in values:
            output.append(float(value))
            
    return output
    
def sendInterface(observations):

    with open("../../fifo_in","w") as fifo:
        output = ";".join([str(x) for x in observations])
        print(output)
        
        fifo.write(output+"\n")

def main():
    
    # create environment
    env=gym.make('CartPole-v1',render_mode='human') 
    
    observation=env.reset()[0]
        
    print("Open fifos")
            
    print("Send data")
    
    reward = 0
    
    terminated=False
    
    to_send = np.zeros(6,dtype=np.float32)
    
    to_send[:4] = observation[:]
    to_send[4] = reward
    
    sendInterface(to_send)
    
    last_observation = np.zeros(4,dtype=np.float32)
    
    steps = 0
    
    log_file = open("log.csv","w")
    
    timestamp = 0
        
    while True:
        
        env.render()
                        
        interface=readInterface()
        
        print(interface)
        
        observation, reward, terminated, truncated, info =env.step(interface[0]>interface[1])
        
        to_send[:4] = observation[:]
        #to_send[4] = ( - (observation[0] ** 2) / 11.52 - (observation[2] ** 2) / 288)
        
        #to_send[4] = -( abs(observation[2]) - abs(last_observation[2]) ) * 10.0
                 
        sendInterface(to_send)
        
        to_send[5] = 0.0
        
        steps +=1
        
        to_send[4] = 0
        
        last_observation[:] = observation[:]
        
        if steps >= 500:
            terminated = True
        
        if terminated:
            timestamp+=1
            observation=env.reset()[0]
            last_observation[:] = observation[:]
            to_send[4] = (steps - 500)/500.0
            to_send[5] = 1.0
            
            log_file.write("{};{}\n".format(timestamp,to_send[4]))
            steps = 0 



if __name__=="__main__":
    main()