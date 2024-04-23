import sysv_ipc as ipc
import gymnasium as gym

import numpy as np

import cartpole_pb2

import os    


def readInterface()->cartpole_pb2.CartPole:
    interface=cartpole_pb2.CartPole()
    fifo=os.open("../../fifo",os.O_RDONLY)
        
    #send_frame(fifo,interface)
    n= os.read(fifo,1)
    if n == b'@':
        size=os.read(fifo,8)
        
        size=int.from_bytes(size,'little')
        
        interface.ParseFromString(os.read(fifo,size))
        
        interface.reward=0.0
            
        os.close(fifo)
        return interface
        
    return None
    
def sendInterface(interface:cartpole_pb2.CartPole):
    fifo=open("../../fifo","wb")
        
    data=interface.SerializeToString()
            
    fifo.write(b'@')
    
    fifo.write(len(data).to_bytes(length=8,byteorder ='little'))
            
    fifo.write(data)
    
    fifo.close()


def main():
    
    # create environment
    env=gym.make('CartPole-v1',render_mode='human') 
    
    observation=env.reset()[0]
    
    interface=cartpole_pb2.CartPole()
    
    interface.reward=11
    
    interface.inputs.extend([1,1,1,1])
    
    interface.outputs.extend([1,1])
    
    print(interface.inputs)
    
    print("Open fifos")
            
    print("Send data")
    
    terminated=False
        
    while True:
        
        env.render()
        
        interface.inputs[0]=observation[0]
        interface.inputs[1]=observation[1]
        interface.inputs[2]=observation[2]
        interface.inputs[3]=observation[3]
        
        #if terminated:
        #    observation=env.reset()[0]
        
        #send observation
        sendInterface(interface)
                
        interface=readInterface()
        
        print(interface.outputs)
        
        observation, reward, terminated, truncated, info =env.step(interface.outputs[0]>interface.outputs[1])
            
        interface.reward=reward
        
        sendInterface(interface)



if __name__=="__main__":
    main()