# Dispatcher Configs
- First, copy `base_experiment_config.json` and name it something meaninful.
- Change flags to your experimental setup. For example, you may want to change the dataset.
- Each arg in search_space is a list. To search over parameter settings, define the values you want to try in the respective lists and the dispatcher will run over all combinations.
- You can change the GPU's you'll use by adding/ removing gpu id's in `available_gpus`
- Run `python scripts/dispatcher.py --config_path=[your_config] --result_path=[path_in_shared_storage_result_dir]`  and the grid search will alert you when it's done!

Please check `num_jobs` logged at the beginning of the program makes sense to make sure you didn't make a mistake in your config file.