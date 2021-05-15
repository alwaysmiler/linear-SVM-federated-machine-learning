import socket
import pickle
from TFClass2_1 import TFML
import numpy as np
ClientSocket = socket.socket()
host = '127.0.0.1'
port = 1233

clientModel=TFML('client2_2')


print('Waiting for connection')
try:
    ClientSocket.connect((host, port))
except socket.error as e:
    print(str(e))
print("connection established")
#Response = ClientSocket.recv(1024)
#(Response.decode('utf-8'))
clientModel.run_init()

new_client_weight=clientModel.model.coef_
new_client_intercept=clientModel.model.intercept_
#print(type(old_client_weight[0]))
while True:

    old_client_weight=new_client_weight
    old_client_intercept = new_client_intercept
    temp=np.concatenate((old_client_weight, np.transpose(np.asarray([old_client_intercept]))), axis=1)
    #print(temp)
    temp2=temp.tolist()
    temp2.append(clientModel.name)
    ClientSocket.send(pickle.dumps(temp2)+b'endingpickle')
    received_weights = b''
    while received_weights[-12:] != b'endingpickle':
        data = ClientSocket.recv(1024)
        received_weights += data
    received_weights = pickle.loads(received_weights[:-12])
    tempnp=np.asarray(received_weights)
    #clientModel.model.set_weights(received_weights)
    clientModel.run(tempnp[:,:-1],tempnp[:,-1])
    clientModel.eval()
    new_client_weight = clientModel.model.coef_
    new_client_intercept = clientModel.model.intercept_


ClientSocket.close()