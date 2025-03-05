import os

# File paths (update these paths as needed)
TRAIN_FILEPATH = os.path.realpath(os.path.join(os.path.curdir, 'content', 'subtaskA_train_monolingual.jsonl'))
DEV_FILEPATH = os.path.realpath(os.path.join(os.path.curdir, 'content', 'subtaskA_dev_monolingual.jsonl'))  # Provided dev set with 5,000 examples (optional)
TEST_FILEPATH = os.path.realpath(os.path.join(os.path.curdir, 'content', 'subtaskA_monolingual.jsonl'))       # Test dataset (without labels)
OUTPUT_FILE = os.path.realpath(os.path.join(os.path.curdir, 'content', 'Result.jsonl'))