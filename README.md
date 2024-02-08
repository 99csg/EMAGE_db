# EMAGE_db

### 1. File Tree
    beat_env
    ├── codes/BEAT
        └── docs
            └── assets (.mp4)
        └── requirements.txt

    ├── datasets/beat_cache
        └── test
            └── bvh_full (.bvh)
            └── bvh_rot (.bvh)
            └── bvh_rot_cache (.mdb)
            └── bvh_rot_vis (.bvh)
            └── emo (.csv)
            └── facial52 (.json)
            └── sem (.txt)
            └── text (.TextGrid)
            └── wave16k (.npy)
        └── train 
            └── bvh_rot (.npy) mean,std
        └── weights (.bin)
        
    └── outputs/audio2pose
        └── wandb   
        └── custom   
            └── 0207_193603_camn_beat_4english_15_141 
                └── .txt
                └── .yaml
                └── 9999 (.bvh)

    └── configs
        └── camn_beat_4english_15_141.yaml

    └── dataloaders
        └── beat.py   
        └── preprocessing.ipynb

    └── models
        └── camn.py   
        └── motion_autoencoder.py   

    └── utils
        └── config.py   
        └── other_tools.py

    └── optimizers   
    
    └── train.py
    └── test.py
    └── ae_trainer.py   
    └── camn_trainer.py   
    └── multi_trainer.py   


### 2. Data description


test_data['pose'] (855,141) 1.2557,0.5165,-1.3564,...

test_data['audio'] (912000) 1.991e-4, 1.03e-1,...

test_data['facial'] (855,51) -0.71, 4.38,...
#### .json 
    ```
    
    └── names : ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",                         "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",                                 "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",                             "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",             z"mouthRollLower",             "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]
    
    └── frames : set { }
        └── weights : 51개 0.71,...
        └── time : 1개 0.81
        └── rotation :[] nothing 
    
    

    ```
test_data['word'] (855) 4,4,4,4,38,50,...,4,4,4

test_data['id'] (1) 

test_data['emo'] (855) 6,6,6,6,6,6,6,6,6,

test_data['sem'] (855) 0.1,0.1,0.7,....0.1

    
    
                       
