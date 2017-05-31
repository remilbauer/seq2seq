"""This will go through a list of sample input files, infer system responses for each utterance
and write new conversation"""
import tensorflow as tf
import os

from bin.infer import main as infer
from bin.infer import FLAGS

chat_mode = True

#Lower the logging level for chat mode
if chat_mode:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


convo_files_dir = "sample_convos/human_utterances"
responses_dir = "sample_convos/ryan"
input_file_base = "human_tick_tock_{}.txt"
output_file_base = "system_context_{}.txt"
infer_yaml = "/ghissubot/model_configs/infer.yml"
model_dir = "trained_models/cornell_context/run2_correct_vocab"

task_args_yaml = \
"""tasks:
  - class: DecodeText
    params:
      unk_replace: True
      output_file: True
      out_filename: fuckme.txt"""

task_args_dict_list = [
    {'class': "DecodeText",
     'params': {'unk_replace': True,
                'output_file': True,
                'out_filename': 'default.txt'}
     }
]

input_args_yaml = \
    """input_pipeline:
class: ConversationInputPipeline
params:
  source_files:
    - temp_input.txt"""


#Set yaml config path in flags
FLAGS.model_dir = model_dir
#Set batch size to 1 since we infer on one utterance at a time
FLAGS.batch_size = 1


def infer_response(utterance, prev_utterance):
    #Make temporary file with concatenated utterance
    full_input = prev_utterance.strip() + '|' + utterance.strip()
    with open("temp_input.txt", 'w', encoding='utf8') as file:
        file.write(full_input + '\n')

    # Set flags for inference
    FLAGS.input_pipeline = input_args_yaml

    # Actually perform inference, so hacky
    infer([])

    tf.reset_default_graph()

    # Now freaking get the models response from out file and return
    with open(task_args_dict_list[0]['params']['out_filename'], 'r', encoding='utf8') as file:
        outputs = file.readlines()

    #Return last response
    return outputs[-1]



## Loop through every input file in directory
if __name__ == "__main__":
    for i in range(10):
        # Setup input and output files
        input_name = convo_files_dir + '/' + input_file_base.format(i)
        task_args_dict_list[0]['params']['out_filename'] = responses_dir + '/' + output_file_base.format(i)
        FLAGS.tasks = task_args_dict_list

        previous_utterance = 'fuck'
        with open(input_name, 'r', encoding='utf8') as file:
            if chat_mode:
                while True:
                    line = input().strip()
                    previous_utterance = infer_response(line, previous_utterance)
            for line in file.readlines():
                previous_utterance = infer_response(line, previous_utterance)
