# federated_iov
This is a project for combining federated learning in IoV and blockchain
First of all it`s better to  build virtual environment and install 
dependencies on it for preventing broken pip:

```commandline
# cd in fderated_iov
$ python3 -m venv venv_iov 
$ source venv_iov/bin/activate
$ pip install -r requirements.txt 
```



<h1>General Structure</h1> 

We have 4 cars or clients or nodes and 2 stations or RSU 
and each station have its data for asses model

<h1>WorkFlow</h1> 
<ul>
<li>First each <b>trainer</b> car train its model  and send it to station using 
<b>flask</b> request to an endpoint for example 'model/send' of its station </li>  

<li>Station stor best model in each iteration  </li>
<li>After a whole round a station find best model base on test accuracy </li>
<li>Each station find best its best model will be responsible for making key block and TCP a request in all network</li>  
<li>All other station send their models to this station</li>
<li>After receiving all the models from all stations the block-maker station make the block and send it to all network</li>
<li>in next round all cars uses the model in the last block in blockchain</li>
</ul>

in order to run a car node use the below command in new terminal
```commandline
# cd in fderated_iov
# put your dataset in federated_iov directory
$ source venv/bin/activate 
$ export ID=20002 && python3 -m src.app.server
```

in order to run a station use below command in new terminal

```commandline
# cd in fderated_iov
$ source venv/bin/activate 
$ export CUDA_VISIBLE_DEVICES="" && export ID=10000 && python3 -m src.app.server
```

in order to run system admin run below command in new terminal

```commandline
# cd in fderated_iov
$ source venv/bin/activate 
$ export CUDA_VISIBLE_DEVICES="" && export ID=5000 && python3 -m src.app.server
```

In your browser search "localhost/5000/make/block" for making a new block 
each time a block is made you will see the massage "block successfully mined"
for see all block info search "localhost/10000/blockchain" for see the length of block 
search "localhost/10000/blockchain/length"

