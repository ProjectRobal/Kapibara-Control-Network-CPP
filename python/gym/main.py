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
    
    to_send = np.zeros(5,dtype=np.float32)
    
    to_send[:4] = observation[:]
    to_send[4] = reward
    
    sendInterface(to_send)
    
    last_observation = np.zeros(4,dtype=np.float32)
        
    while True:
        
        env.render()
                        
        interface=readInterface()
        
        print(interface)
        
        observation, reward, terminated, truncated, info =env.step(interface[0]>interface[1])
        
        to_send[:4] = observation[:]
        #to_send[4] = ( - (observation[0] ** 2) / 11.52 - (observation[2] ** 2) / 288) * 100.0
        
        to_send[4] = -( abs(observation[2]) - abs(last_observation[2]) ) * 100.0
                 
        sendInterface(to_send)
        
        last_observation[:] = observation[:]
        
        if terminated:
            observation=env.reset()[0]
            last_observation[:] = observation[:]



if __name__=="__main__":
    main()