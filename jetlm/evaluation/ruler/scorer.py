# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Get summary.csv with score and null predictions amount.

Running
```
python evaluate.py \
    --data_dir /path/to/your/prediction_jsonl_folder \
    --benchmark synthetic
```
"""

import re
import os
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

from tqdm import tqdm


def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def get_pred_and_ref(
    outputs: list[dict],
    task_config: dict,
    input_field: str = 'input',
    references_field: str = 'outputs',
    prediction_field: str = 'pred',
    metadata_field: str = 'others',
):

    inputs = []
    predicts = []
    references = []
    indices = []

    for line in tqdm(outputs, desc="Geting preds and refs"):
        input = line[input_field]
        predict = line[prediction_field]
        predict = postprocess_pred(predict, task_config)
        reference = line.get(references_field, [line.get('output', '')])
        index = line[metadata_field].get('id', line['idx'])
        
        inputs.append(input)
        predicts.append(predict)
        references.append(reference)
        indices.append(index)
        
    return inputs, predicts, references, indices


def run_evaluation_per_task(
    task_config: dict, outputs: list[dict], verbose: int = 0):
    inputs, predicts, references, indices = get_pred_and_ref(
        outputs,
        task_config=task_config,
    )

    task_nulls = f'{sum([len(x)==0 for x in predicts])}/{len(predicts)}'

    if len(references) > 0 and references[0][0] is not None:
        task_score = task_config['metric_fn'](predicts, references)
    else:
        task_score = 0.0

    if verbose != 0:
        print('=' * 40)
        for i, (input, reference, predict) in enumerate(zip(inputs, references, predicts)):
            print(f'Input     : {input}')
            print(f'Reference : {reference}')
            print(f'Prediction: {predict}')
            print('=' * 40)
            if i > verbose:
                break

    return task_score, task_nulls, predicts, indices
