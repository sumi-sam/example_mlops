import os
import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import get_data,read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import ElasticNet
import joblib
import json
import numpy as np

def eval_metrics(actual,predicted):
    rmse=np.sqrt(mean_squared_error(actual,predicted))
    mae=mean_absolute_error(actual,predicted)
    r2=r2_score(actual,predicted)
    return rmse,mae,r2



def train_and_evaluate(config_path):
    config=read_params(config_path)
    test_data_path=config["split_data"]["test_path"]
    train_data_path=config["split_data"]["train_path"]
    random_state=config["base"]["random_state"]
    model_dir=config["model_dir"]
    alpha=config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio=config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target=[config["base"]["target_col"]]
    train =pd.read_csv(train_data_path,sep=",",encoding="utf-8")
    test =pd.read_csv(test_data_path,sep=",",encoding="utf-8")
    
    Train_x=train.drop(target,axis=1)
    Test_x=test.drop(target,axis=1)
    Train_y=train[target]
    Test_y=test[target]
    ######################################

    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=random_state)
    lr.fit(Train_x,Train_y)

    predicted_qualities=lr.predict(Test_x)

    (rmse,mae,r2)=eval_metrics(Test_y,predicted_qualities)
    print('RMSE=',rmse)
    print('mae=',mae)
    print('R2=',r2)
    score_file=config["reports"]["scores"]
    params_file=config["reports"]["params"]

    with open(score_file,"w") as f:
        score={
            "rmse":rmse,
            "mae":mae,
            "r2":r2
        }
        json.dump(score,f,indent=4)

    with open(params_file,"w") as f:
        params={
            "alpha":alpha,
            "l1_ratio":l1_ratio
        }
        json.dump(params,f,indent=4)
    
    os.makedirs(model_dir,exist_ok=True)

    model_path=os.path.join(model_dir,"model.joblib")
    joblib.dump(lr,model_path)
    return train_and_evaluate

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

