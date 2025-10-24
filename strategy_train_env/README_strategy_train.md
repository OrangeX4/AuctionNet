# Overview
This is an auto-bidding strategy training module to help users implement and evaluate their bidding strategies. This framework includes three modules: data processing, strategy training, and offline evaluation. Several industry-proven baseline strategies, such as reinforcement learning-based bidding, online linear programming-based bidding and generative models, are included in the framework. Users can utilize this framework to develop a well-trained auto-bidding strategy based on the training dataset. Users can also rely on the provided framework for a basic offline assessment to evaluate the strategy.




## Data Processing
Download the raw data on ad opportunities granularity and place it in the biddingTrainENv/data/ folder.
The directory structure under data should be:
```
NeurIPS_Auto_Bidding_General_Track_Baseline
|── data
    |── traffic
        |── period-7.csv
        |── period-8.csv
        |── period-9.csv
        |── period-10.csv
        |── period-11.csv
        |── period-12.csv
        |── period-13.csv
        |── period-14.csv
        |── period-15.csv
        |── period-16.csv
        |── period-17.csv
        |── period-18.csv
        |── period-19.csv
        |── period-20.csv
        |── period-21.csv
        |── period-22.csv
        |── period-23.csv
        |── period-24.csv
        |── period-25.csv
        |── period-26.csv
        |── period-27.csv
        
```

Run this script to convert the raw data on ad opportunities granularity into trajectory data required for model training.
```
python  bidding_train_env/train_data_generator/train_data_generator.py
```

## strategy training
### reinforcement learning-based bidding

#### IQL(Implicit Q-learning) Model
Load the training data and train the IQL bidding strategy.
```
python run/run_iql.py 
```

#### BC(behavior cloning) Model
Load the training data and train the BC bidding strategy.
```
python run/run_bc.py 
```

#### BCQ  Model
Load the training data and train the BCQ bidding strategy.
```
python run/run_bcq.py 
```

#### CQL  Model
Load the training data and train the CQL bidding strategy.
```
python run/run_cql.py 
```

#### TD3_BC  Model
Load the training data and train the TD3_BC bidding strategy.
```
python run/run_td3_bc.py 
```


### online linear programming-based bidding
#### OnlineLp Model
Load the training data and train the OnlineLp bidding strategy.
```
python run/run_onlinelp.py 
```

### Generative Model
#### Decision-Transformer
Load the training data and train the DT bidding strategy.
```
python run/run_decision_transformer.py 
```




## offline evaluation
Load the raw data on ad opportunities granularity to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python run/run_evaluate.py --algorithm <algorithm>
```
For example, to evaluate the IQL strategy:
```
python run/run_evaluate.py --algorithm iql
```

```
python run/run_evaluate.py --help 
usage: run_evaluate.py [-h] [--algo {iql,bc,bcq,cql,td3_bc,onlinelp,dt}]

Run offline evaluation for bidding strategies.

options:
  -h, --help            show this help message and exit
  --algo {iql,bc,bcq,cql,td3_bc,onlinelp,dt}
                        The bidding algorithm to evaluate.
```