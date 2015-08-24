all:
	echo "fold, featu, [model, opt], [ensemble, subm], show"

fold:
	python gen_fold.py

featu:
	python gen_feat.py

model:
	python model_library.py "train" | tee -a log/model.log

opt:
	python model_library.py "hyperopt" | tee -a log/hyperopt.log

ensemble:
	python gen_ensemble.py "ensemble" | tee -a log/ensemble.log

subm:
	python gen_ensemble.py "submission" | tee -a log/subm.log

show:
	python utils.py
