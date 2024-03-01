from dpo.core import MeetingDialog
import random
import pandas as pd
from loguru import logger
logger.add("main_errors.log")
epochs = 1000
final_results_df = pd.DataFrame()
# baseline run
topics = [
    "generative ai",
    "hackathon",
    "google gen AI",
    "teaching",
    "education",
    "social media",
    "bubble tea",
    "the social impact of barbie girls",
    "parabola",
    "how to train a model",
    "how awesome huggingface is",
    "politics"
]
min_max_participants = (2, 3)
min_max_talk_turns = (4, 8)
file_name = "baseline_run.xlsx"

for epoch in range(epochs):
    topic = random.choice(topics)
    md = MeetingDialog(
       num_participants=random.randint(*min_max_participants),
       talk_turns_before_intervention=random.randint(*min_max_talk_turns),
       topic=topic
    )
    try:
        result = md.run()
    except Exception as e:
        logger.error(f"MeetingDialog.run() failed with reason: {e}")
    
    epoch_result_df = pd.DataFrame({
        'meeting_id': [epoch]*len(result),
        'type': ['baseline']*len(result),
        'conversation': result
    })
    final_results_df = pd.concat([final_results_df, epoch_result_df])

    if epoch % 10 == 0 or epoch == epochs - 1:
        final_results_df.to_excel(file_name)

