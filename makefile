
.PHONY preprocessed_data
preprocessed_data: 
	python3 preprocessing/preprocess.py

data/256_ObjectCategories: preprocessing/preprocess.py
	python3 preprocessing/preprocess.py

data/train: preprocessing/preprocess.py
	python3 preprocessing/preprocess.py

data/test: preprocessing/preprocess.py
	python3 preprocessing/preprocess.py

.PHONY clean_data

clean_data:
	rm -rf data/*
