# Thank You for Checking it out 

# As the first step, you need to create a conda environement
	conda update conda
	conda env create -f environment.yml
	conda activate ngf 

# Creation of Benign and Backdoor models
For creating a Benign model-  
	
	python benign_model.py --output-dir defined/save/folder/ 

For Creating a backdoor model-
	
	python backdoor_model.py --poison-type blend --poison-rate 0.10 --output-dir defined/save/folder/ 


# Purifying a Model
	python Purification_NGF.py --poison-type blend --saved_models defined/save/folder/
