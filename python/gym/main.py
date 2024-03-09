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
    
    print(len(data).to_bytes(length=8))
        
    fifo.write(data)
    
    fifo.close()


def main():
    
    
    interface=cartpole_pb2.CartPole()
    
    interface.reward=11
    
    interface.inputs.extend([1,1,1,1])
    
    interface.outputs.extend([1,1])
    
    print(interface.inputs)
    
    print("Open fifos")
            
    print("Send data")
        
    while True:
        
        sendInterface(interface)
        
        print("Reading outputs")
        
        interface=readInterface()
        
        sendInterface(interface)



if __name__=="__main__":
    main()