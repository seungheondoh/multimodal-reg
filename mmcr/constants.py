import os
import json

NOW_CHANNEL = ['ARTIS', 'DRAMA', 'ENTER', 'NEWS', 'SPORT']
CREATOR = json.load(open(os.path.join("../../dataset", "now","balance","train_channel.json"), 'r'))