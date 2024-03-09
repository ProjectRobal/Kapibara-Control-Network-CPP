import sysv_ipc as ipc
import gymnasium as gym

import numpy as np

import cartpole_pb2

import os    


def main():
    
    
    interface=cartpole_pb2.CartPole()
    
    interface.reward=0
    
    print("Open fifos")
    
    #fifo=open("../../fifo","rb")
    
    fifo=open("../../fifo","wb")
    
    data=interface.SerializeToString()
    
    print("Sending data")
    
    fifo.write(b'@')
    
    fifo.write(len(data).to_bytes())
    
    fifo.write(data)
    
    fifo.close()
    
    print("Send data")
        
    while True:
        
        print("Reading input")
        fifo=os.open("../../fifo",os.O_RDONLY)
        
        #send_frame(fifo,interface)
        n= os.read(fifo,1)
        print(n)        
        if n == b'@':
            size=os.read(fifo,8)
            
            size=int.from_bytes(size,'little')
            print(size)
            
            interface.ParseFromString(os.read(fifo,size))
            print(interface.reward)
            
            interface.reward=0.0
            
        os.close(fifo)
        
        fifo=open("../../fifo","wb")
        
        fifo.write(b'@')
    
        fifo.write(len(data).to_bytes())
        
        fifo.write(data)
        
        fifo.close()



if __name__=="__main__":
    main()