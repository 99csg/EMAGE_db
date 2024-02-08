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


#### test_data['pose'] (855,141) 1.2557,0.5165,-1.3564,...
##### .bvh - motion capture dataset

#### test_data['audio'] (912000) 1.991e-4, 1.03e-1,...
##### .npy w

#### test_data['facial'] (855,51) -0.71, 4.38,...
##### .json     
    └── names : ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight","eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",                 "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",                            "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",             z"mouthRollLower","mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]
    
    └── frames : set { }
        └── weights : 51개 0.71,...
        └── time : 1개 0.81
        └── rotation :[] nothing 
    
    
    
#### test_data['word'] (855) 4,4,4,4,38,50,...,4,4,4
##### .TextGrid  
	└── File type = "ooTextFile"
        Object class = "TextGrid"
        
        xmin = 0.0
        xmax = 68
        tiers? <exists>
        size = 2 # two types of tiers - ex) words, phonemes
        item []:
        	item [1]: # tier 1 - IntervalTier 
        		class = "IntervalTier"
        		name = "words"
        		xmin = 0.0
        		xmax = 68
                intervals: size = 213
        			intervals [1]:
        				xmin = 0.0  # timeline 0s~1.47s
        				xmax = 1.47
        				text = ""
        			intervals [2]:
        				xmin = 1.47
        				xmax = 2.36
        				text = "well"
        			intervals [3]:
        				xmin = 2.36
        				xmax = 2.56
        				text = ""
        			intervals [4]:
        				xmin = 2.56
        				xmax = 3.05
        				text = "in"
#### test_data['id'] (1) 

#### test_data['emo'] (855) 6,6,6,6,6,6,6,6,6,
##### .csv  
    └── 00_netural, 0, 63.908, 63.908, 0

#### test_data['sem'] (855) 0.1,0.1,0.7,....0.1
##### .txt
	                start(s) end(s) duration(s) score descriptors
    └── 01_beat_align	0	3.119	3.119	0.1
	07_iconic_h	3.119	4.785	1.666	0.7	relaxing
	01_beat_align	4.785	6.514	1.729	0.1
	06_iconic_m	6.514	7.389	0.875	0.6	tired
	01_beat_align	7.389	7.744	0.355	0.1
	05_iconic_l	7.744	8.369	0.625	0.5	started
	01_beat_align	8.369	13.494	5.125	0.1
	10_metaphoric_h	13.494	15.358	1.864	1.0	Monday through Friday
	01_beat_align	15.358	20.369	5.011	0.1
	05_iconic_l	20.369	21.181	0.812	0.5	complain
	01_beat_align	21.181	21.389	0.208	0.1
	05_iconic_l	21.389	22.494	1.105	0.5	completing
	01_beat_align	22.494	24.16	1.666	0.1
	09_metaphoric_m	24.16	25.181	1.021	0.9	Okay
	01_beat_align	25.181	29.389	4.208	0.1
	02_deictic_l	29.389	30.494	1.105	0.2	friends
	01_beat_align	30.494	35.16	4.666	0.1
	09_metaphoric_m	35.16	36.181	1.021	0.9	sunshine
	01_beat_align	36.181	43.202	7.021	0.1
	05_iconic_l	43.202	44.844	1.642	0.5	working
	01_beat_align	44.844	63.908	19.064	0.1


    
    
                       
