# Anonymous

## Fine-Tune

This is a simulation code.

Due to the policy of LLaMA, this project will not provide LLaMA and Alpaca models. Users need to apply for the LLaMA model on the official website of LLaMA.

Below is an example method for starting fine-tuning. The model obtained by the user should be placed in the "model_bak" folder, and the fine-tuning data should be placed in "data_generation/data_humaneval/only_python". The output of fine-tuning results will be stored in "./python_humaneval". There are a total of 10 organizations participating in fine-tuning, with each organization training one round per communication round locally. At the same time, an adapter is centrally fine-tuned using all data for comparison purposes.

python main.py --global_model
'./model_bak'      --data_path  "./data_generation/data_humaneval/only_python"       --output_dir  './python_humaneval/'      --num_communication_rounds 3       --num_clients  10
--local_num_epochs 1    --train_on_inputs False    --client_selection_frac 1.0     --group_by_length  --central True

We put the adapter we fine-tuned into the "adapter" folder.

Our experimental data is stored in the "exp_result" folder.


## Evaluate

We provide the files we use for model validation. Taking the humaneval task as an example, our validation file is named eval_humaneval.py. The code for other tasks is similar to this file. There are three places that need to be modified in the file:

adapter_url = r"adapter/python_humaneval/10"

base_model = r"/model_bak"

write_jsonl("new_FL_Alpaca_Sum_1.jsonl", samples)

The adapter_url represents the path of the adapter.

The base_model represents the path of the base model.

The write_jsonl function indicates where to output the results.

We also provide a file for manual model validation, where users manually input questions. The file is named eval_manually.py and requires modifications to adapter_url and base_model.
