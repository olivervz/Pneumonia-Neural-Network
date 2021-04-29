clean:
	-rm model/model.h5 model/model.json 
	-rm -r main/__pycache__
removeVENV:
	-rm -r env
load:
	python3 main/load.py
train:
	python3 main/train.py
run:
	python3 main/run.py
