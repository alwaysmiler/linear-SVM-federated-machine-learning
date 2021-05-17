# Federated-Machine-Learning-Model-
1. Install 64 bit Python V3.8
2. Install tensorflow and other required package
3. Change the data directory where all data stored in line 13 in both TFClass.py and TFClass_1.py datafolder = r'C:\Users\tingx\Downloads\MeasurementsFromAgnesDevices\Agnes-2' , TFClass2.py and TFClass2_1.py datafolder = r'C:\Users\tingx\Downloads\MeasurementsFromAgnesDevices\Agnes-3'
4. Run Server.py-->it will output the weights update for all the clients include client 1, client 1_1, client 2, and client2_2 for each iteration
5. Run Client1.py, Client1_1.py, Client2.py, Client2_2.py-> it will output the accuracy for each client.
6. Currently the stop criteria has not be been set up. It will continue to run. Final accuracy will be approaching 1 eventually and weights converge (weights do not change at local client and the weights of two client are close to each other).

