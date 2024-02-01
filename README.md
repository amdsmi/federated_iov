# federated_iov
This is a project for combining federated learning in IoV and blockchain

First of all it`s better to  build virtual environment and install dependencies on it:

```commandline
$ python3 -m venv venv_iov 
$ source venv_iov/bin/activate
$ pip install -r requirements.txt 
```



<h1>General Structure</h1> 

We have 12 cars or clients or nodes and 3 stations or RSU each station have 4 cars so that 3 of them are 
trainers and 1 of them is data collector. We have micro-chain made of micro blocks between each station cars
and its station that have local-global-aggregated models 

<h1>WorkFlow</h1> 
<ul>
<li>First each <b>trainer</b> car train its model  and send it to station using <b>flask</b> request to an endpoint for example 'model/send'</li>  
<li>Data Collector car send its data using request with <b>flask</b> too </li>
<li>Stations give this data and store it in class or somthing like that </li>
<li>So sending the models and data just handles with flask</li>
<li>Stations aggregate this There model using <b>Genetic Algorithm</b> and stor it in a micro-block on micro-chaine</li>  
<li>After 10 rounds and putting 10 models on micro-chaine stations start to wining and writing the global model in key block</li>
<li>Each station could write key block <b>publish</b> the updated chain for all cars and stations so all station and cars should be on same network using pubnub</li>
</ul>
