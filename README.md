NeuralNetworkSimulator
==================

NeuralNetworkSimulator is the final project of the Machine Learning course, whose goal was to design and implement a multilayer perceptron with one layer of fully connected hidden units and its online back-propagation algorithm from scratch. The hyper-parameter selection was performed by k-fold cross-validation and the neural network was tested both on [classification](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems) and regression tasks.


Setting up
------------

To run the NeuralNetworkSimulator, the Monk and Loc datasets are needed in the same root of the NeuralNetworkSimulator directory.
If the NeuralNetworkSimulator path is `~/NeuralNetworkSimulator/` then the dataset path has to be `~/data/monk/` for the Monk dataset and `~/data-AA1-2013-CUP/` for the Loc dataset.
To see the results of the NeuralNetworkSimulator you need to create the results directory: if the NeuralNetworkSimulator path is the same as before then the results directory has to be `~/results/`

Options
---------


	--monk N:	runs the NeuralNetworkSimulator on the Monk N
				dataset doing the simple validation

	--loc:		runs the NeuralNetworkSimulator on the Loc
				dataset doing the 5-folds and 10-folds cross-validation

If you don’t want to do the model selection you can specify the neural network hyper-parameters:

	--nNeurons int:		sets the number of neurons of the neural network
	--epochs int:		sets the number of epochs
	--eta double:		sets eta
	--alpha double:		sets alpha
	--lambda double:	sets lambda

To do simple runs, after `--loc`, you can use:

	--run:	runs the NeuralNetworkSimulator using the 70%
			of the modeling data (70% of the training data)
			for the training and 30% for the validation.

	--fold:	runs the NeuralNetworkSimulator splitting the 70%
			of the modeling data into 10 folds.
			One fold is used for the validation and the remaining 
			for the training. 

Usage Examples
-------------------

	--monk 1:	runs the NeuralNetworkSimulator on Monk 1 
				dataset doing the simple validation

	--monk 2 --nNeurons 5 --epochs 500 --eta 0.05 --alpha 0.2 --lambda 0.08:
				runs the NeuralNetworkSimulator on the Monk 2 dataset
				with the hyper-parameters specified

	--loc --nNeurons 15 --epochs 10000 --eta 0.005 --alpha 0.5 --lambda 0.001:
				runs the NeuralNetworkSimulator on the Loc
				dataset with the hyper-parameters specified
