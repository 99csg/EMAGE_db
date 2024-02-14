# EMAGE_db

### 1. File Tree



### 2. Data description


#### 2-1. test_data['pose'] (855,141) 1.2557,0.5165,-1.3564,...
.bvh(motion capture dataset) -emage=smplxflame_30

#### 2-2. test_data['audio'] (912000) 1.991e-4, 1.03e-1,...
.npy -emage==

#### 2-3. test_data['facial'] (855,51) -0.71, 4.38,...
.json -emage=smplxflame_30

	names : ["browDownLeft", ..., "noseSneerRight"]
	frames : set { }
        └── weights : 51개 0.71,...
        └── time : 1개 0.81
        └── rotation :[] nothing 
    
#### 2-4. test_data['word'] (855) 4,4,4,4,38,50,...,4,4,4
.TextGrid -emage==
 	
        File type = "ooTextFile"
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
	    			...
        			intervals [4]:
        				xmin = 2.56
        				xmax = 3.05
        				text = "in"
#### 2-5. test_data['id'] (1) 

#### 2-6. test_data['emo'] (855) 6,6,6,6,6,6,6,6,6,
.csv  -emage==

     00_netural, 0, 63.908, 63.908, 0

#### 2-7. test_data['sem'] (855) 0.1,0.1,0.7,....0.1
.txt -emage==

	                start(s) end(s) duration(s) score descriptors
    	01_beat_align	0	3.119	3.119	0.1
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



### 3. model architecture(EMAGE)

test_demo.py 
input : audio, text
output : motion 

	  (module): MAGE_Transformer(
	    (text_pre_encoder_face): Embedding(11195, 300)
	    (text_encoder_face): Linear(in_features=300, out_features=256, bias=True)
	    (text_pre_encoder_body): Embedding(11195, 300)
	    (text_encoder_body): Linear(in_features=300, out_features=256, bias=True)
	    (audio_pre_encoder_face): WavEncoder(
	      (feat_extractor): Sequential(
	        (0): BasicBlock(
	          (conv1): Conv1d(2, 64, kernel_size=(15,), stride=(5,), padding=(1600,))
	          (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(2, 64, kernel_size=(15,), stride=(5,), padding=(1600,))
	            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	        (1): BasicBlock(
	          (conv1): Conv1d(64, 64, kernel_size=(15,), stride=(6,))
	          (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(64, 64, kernel_size=(15,), stride=(6,))
	            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	        (2): BasicBlock(
	          (conv1): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	        )
	        (3): BasicBlock(
	          (conv1): Conv1d(64, 128, kernel_size=(15,), stride=(6,))
	          (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(128, 128, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(64, 128, kernel_size=(15,), stride=(6,))
	            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	        (4): BasicBlock(
	          (conv1): Conv1d(128, 128, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(128, 128, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	        )
	        (5): BasicBlock(
	          (conv1): Conv1d(128, 256, kernel_size=(15,), stride=(3,))
	          (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(256, 256, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(128, 256, kernel_size=(15,), stride=(3,))
	            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	      )
	    )
	    (audio_pre_encoder_body): WavEncoder(
	      (feat_extractor): Sequential(
	        (0): BasicBlock(
	          (conv1): Conv1d(2, 64, kernel_size=(15,), stride=(5,), padding=(1600,))
	          (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(2, 64, kernel_size=(15,), stride=(5,), padding=(1600,))
	            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	        (1): BasicBlock(
	          (conv1): Conv1d(64, 64, kernel_size=(15,), stride=(6,))
	          (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(64, 64, kernel_size=(15,), stride=(6,))
	            (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	        (2): BasicBlock(
	          (conv1): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	        )
	        (3): BasicBlock(
	          (conv1): Conv1d(64, 128, kernel_size=(15,), stride=(6,))
	          (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(128, 128, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(64, 128, kernel_size=(15,), stride=(6,))
	            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	        (4): BasicBlock(
	          (conv1): Conv1d(128, 128, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(128, 128, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	        )
	        (5): BasicBlock(
	          (conv1): Conv1d(128, 256, kernel_size=(15,), stride=(3,))
	          (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act1): LeakyReLU(negative_slope=0.01, inplace=True)
	          (conv2): Conv1d(256, 256, kernel_size=(15,), stride=(1,), padding=(7,))
	          (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          (act2): LeakyReLU(negative_slope=0.01, inplace=True)
	          (downsample): Sequential(
	            (0): Conv1d(128, 256, kernel_size=(15,), stride=(3,))
	            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	          )
	        )
	      )
	    )
	    (at_attn_face): Linear(in_features=512, out_features=512, bias=True)
	    (at_attn_body): Linear(in_features=512, out_features=512, bias=True)
	    (motion_encoder): VQEncoderV6(
	      (main): Sequential(
	        (0): Conv1d(337, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): ResBlock(
	          (model): Sequential(
	            (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	            (1): LeakyReLU(negative_slope=0.2, inplace=True)
	            (2): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	          )
	        )
	        (3): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	        (4): LeakyReLU(negative_slope=0.2, inplace=True)
	        (5): ResBlock(
	          (model): Sequential(
	            (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	            (1): LeakyReLU(negative_slope=0.2, inplace=True)
	            (2): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	          )
	        )
	        (6): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	        (7): LeakyReLU(negative_slope=0.2, inplace=True)
	        (8): ResBlock(
	          (model): Sequential(
	            (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	            (1): LeakyReLU(negative_slope=0.2, inplace=True)
	            (2): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
	          )
	        )
	      )
	    )
	    (feature2face): Linear(in_features=512, out_features=768, bias=True)
	    (face2latent): Linear(in_features=768, out_features=256, bias=True)
	    (transformer_de_layer): TransformerDecoderLayer(
	      (self_attn): MultiheadAttention(
	        (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	      )
	      (multihead_attn): MultiheadAttention(
	        (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	      )
	      (linear1): Linear(in_features=768, out_features=1536, bias=True)
	      (dropout): Dropout(p=0.1, inplace=False)
	      (linear2): Linear(in_features=1536, out_features=768, bias=True)
	      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	      (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	      (dropout1): Dropout(p=0.1, inplace=False)
	      (dropout2): Dropout(p=0.1, inplace=False)
	      (dropout3): Dropout(p=0.1, inplace=False)
	    )
	    (face_decoder): TransformerDecoder(
	      (layers): ModuleList(
	        (0-3): 4 x TransformerDecoderLayer(
	          (self_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (multihead_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (linear1): Linear(in_features=768, out_features=1536, bias=True)
	          (dropout): Dropout(p=0.1, inplace=False)
	          (linear2): Linear(in_features=1536, out_features=768, bias=True)
	          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (dropout1): Dropout(p=0.1, inplace=False)
	          (dropout2): Dropout(p=0.1, inplace=False)
	          (dropout3): Dropout(p=0.1, inplace=False)
	        )
	      )
	    )
	    (position_embeddings): PeriodicPositionalEncoding(
	      (dropout): Dropout(p=0.1, inplace=False)
	    )
	    (transformer_en_layer): TransformerEncoderLayer(
	      (self_attn): MultiheadAttention(
	        (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	      )
	      (linear1): Linear(in_features=768, out_features=1536, bias=True)
	      (dropout): Dropout(p=0.1, inplace=False)
	      (linear2): Linear(in_features=1536, out_features=768, bias=True)
	      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	      (dropout1): Dropout(p=0.1, inplace=False)
	      (dropout2): Dropout(p=0.1, inplace=False)
	    )
	    (motion_self_encoder): TransformerEncoder(
	      (layers): ModuleList(
	        (0): TransformerEncoderLayer(
	          (self_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (linear1): Linear(in_features=768, out_features=1536, bias=True)
	          (dropout): Dropout(p=0.1, inplace=False)
	          (linear2): Linear(in_features=1536, out_features=768, bias=True)
	          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (dropout1): Dropout(p=0.1, inplace=False)
	          (dropout2): Dropout(p=0.1, inplace=False)
	        )
	      )
	    )
	    (audio_feature2motion): Linear(in_features=256, out_features=768, bias=True)
	    (feature2motion): Linear(in_features=256, out_features=768, bias=True)
	    (bodyhints_face): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=256, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=256, bias=True)
	      )
	    )
	    (bodyhints_body): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=256, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=256, bias=True)
	      )
	    )
	    (motion2latent_upper): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=768, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=768, bias=True)
	      )
	    )
	    (motion2latent_hands): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=768, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=768, bias=True)
	      )
	    )
	    (motion2latent_lower): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=768, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=768, bias=True)
	      )
	    )
	    (wordhints_decoder): TransformerDecoder(
	      (layers): ModuleList(
	        (0-7): 8 x TransformerDecoderLayer(
	          (self_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (multihead_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (linear1): Linear(in_features=768, out_features=1536, bias=True)
	          (dropout): Dropout(p=0.1, inplace=False)
	          (linear2): Linear(in_features=1536, out_features=768, bias=True)
	          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (dropout1): Dropout(p=0.1, inplace=False)
	          (dropout2): Dropout(p=0.1, inplace=False)
	          (dropout3): Dropout(p=0.1, inplace=False)
	        )
	      )
	    )
	    (upper_decoder): TransformerDecoder(
	      (layers): ModuleList(
	        (0): TransformerDecoderLayer(
	          (self_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (multihead_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (linear1): Linear(in_features=768, out_features=1536, bias=True)
	          (dropout): Dropout(p=0.1, inplace=False)
	          (linear2): Linear(in_features=1536, out_features=768, bias=True)
	          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (dropout1): Dropout(p=0.1, inplace=False)
	          (dropout2): Dropout(p=0.1, inplace=False)
	          (dropout3): Dropout(p=0.1, inplace=False)
	        )
	      )
	    )
	    (hands_decoder): TransformerDecoder(
	      (layers): ModuleList(
	        (0): TransformerDecoderLayer(
	          (self_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (multihead_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (linear1): Linear(in_features=768, out_features=1536, bias=True)
	          (dropout): Dropout(p=0.1, inplace=False)
	          (linear2): Linear(in_features=1536, out_features=768, bias=True)
	          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (dropout1): Dropout(p=0.1, inplace=False)
	          (dropout2): Dropout(p=0.1, inplace=False)
	          (dropout3): Dropout(p=0.1, inplace=False)
	        )
	      )
	    )
	    (lower_decoder): TransformerDecoder(
	      (layers): ModuleList(
	        (0): TransformerDecoderLayer(
	          (self_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (multihead_attn): MultiheadAttention(
	            (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
	          )
	          (linear1): Linear(in_features=768, out_features=1536, bias=True)
	          (dropout): Dropout(p=0.1, inplace=False)
	          (linear2): Linear(in_features=1536, out_features=768, bias=True)
	          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
	          (dropout1): Dropout(p=0.1, inplace=False)
	          (dropout2): Dropout(p=0.1, inplace=False)
	          (dropout3): Dropout(p=0.1, inplace=False)
	        )
	      )
	    )
	    (face_classifier): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=256, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=256, bias=True)
	      )
	    )
	    (upper_classifier): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=256, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=256, bias=True)
	      )
	    )
	    (hands_classifier): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=256, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=256, bias=True)
	      )
	    )
	    (lower_classifier): MLP(
	      (mlp): Sequential(
	        (0): Linear(in_features=256, out_features=768, bias=True)
	        (1): LeakyReLU(negative_slope=0.2, inplace=True)
	        (2): Linear(in_features=768, out_features=256, bias=True)
	      )
	    )
	    (motion_down_upper): Linear(in_features=768, out_features=256, bias=True)
	    (motion_down_hands): Linear(in_features=768, out_features=256, bias=True)
	    (motion_down_lower): Linear(in_features=768, out_features=256, bias=True)
	    (spearker_encoder_body): Embedding(25, 768)
	    (spearker_encoder_face): Embedding(25, 768)
	  )
	)

    
    
                       
