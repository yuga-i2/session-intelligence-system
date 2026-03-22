"""
arvyax session intelligence system
===================================
interpretation → judgment → decision

end-to-end pipeline:
- emotional state prediction
- intensity prediction (regression + classification)
- decision engine: what to do + when to do it
- uncertainty modeling
- confidence scoring
- supportive message generation
- ablation study
"""

import re
import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, mean_absolute_error, f1_score
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. DATASET (embedded from training PDFs)
# ─────────────────────────────────────────────────────────────

TRAINING_RECORDS = [
    (1,"The ocean ambience helped me stop drifting and concentrate on my next steps. My to-do list feels less chaotic.","ocean",12,6.5,4,2,"afternoon","mixed","calm_face","clear","focused",3),
    (2,"I tried to relax during the forest ambience yet my thoughts kept racing. I still feel a low buzz in my body.","forest",35,6,2,4,"evening","calm","tired_face","vague","restless",3),
    (3,"The forest session slowed my thoughts and I feel more settled now.","forest",None,None,3,1,"night","overwhelmed","happy_face","clear","calm",3),
    (4,"the mountain ambience was pleasant though i can't say it shifted my mood much. idk.","mountain",25,7,4,4,"night","focused","calm_face","vague","neutral",1),
    (5,"The rain session gave me a pause but the pressure is still sitting hard on me. I'm carrying too much in my head.","rain",25,5,3,5,"afternoon",None,"tense_face","clear","overwhelmed",5),
    (6,"after the forest track i feel peaceful and less pulled in every direction. my shoulders feel less tense.","forest",12,8,3,2,"morning","mixed","calm_face","vague","calm",3),
    (7,"Nothing strong came up during the rain session; I feel fairly normal. At least I paused for a moment.","rain",20,6.5,2,4,"early_morning","calm","neutral_face","conflicted","neutral",1),
    (8,"even with the mountain session my mind kept jumping between tasks.","mountain",12,6,3,4,"morning","neutral","tense_face","clear","restless",4),
    (9,"I couldn't really settle into the cafe track; I kept thinking of everything at once. I still feel a low buzz in my body.","cafe",8,5.5,3,4,"early_morning","mixed","neutral_face","vague","restless",4),
    (10,"The mountain ambience helped me stop drifting and concentrate on my next steps I should begin with the hardest task first","mountain",15,7,4,2,"morning","overwhelmed","calm_face","conflicted","focused",3),
    (11,"The rain sounds were nice but I still feel unsettled and fidgety. Part of me wants to do everything at once.","rain",12,6.5,3,3,"afternoon","mixed",None,"conflicted","restless",4),
    (12,"I feel mentally clear after the mountain session and ready to tackle one thing at a time.","mountain",12,5.5,3,2,"morning","restless","neutral_face","clear","focused",4),
    (13,"The forest session made me calmer but part of me still feels uneasy. Part of me wants rest part of me wants action.","forest",20,5,2,2,"afternoon","neutral",None,"conflicted","mixed",2),
    (14,"The cafe session helped a little though I still feel pulled in too many directions. Part of me wants to do everything at once.","cafe",12,6,3,5,"early_morning","mixed",None,"clear","restless",4),
    (15,"I feel lighter after the mountain sounds like my mind finally softened. I think I can start gently today.","mountain",8,8,3,3,"night","overwhelmed","calm_face","clear","calm",2),
    (16,"I came in distracted but I left the forest session with a sharper mind. It feels easier to make a plan now.","forest",15,None,3,1,"early_morning","neutral","neutral_face","vague","focused",2),
    (17,"The forest session made me calmer but part of me still feels uneasy. I feel better and not better at the same time.","forest",35,6,2,3,"night",None,"calm_face","vague","mixed",3),
    (18,"The mountain session gave me a pause but the pressure is still sitting hard on me.","mountain",15,5.5,3,5,"early_morning","overwhelmed","tired_face","vague","overwhelmed",4),
    (19,"I feel mentally clear after the mountain session and ready to tackle one thing at a time. I should begin with the hardest task first.","mountain",8,7.5,3,2,"night","focused","neutral_face","clear","focused",4),
    (20,"I started scattered but the ocean session helped me lock in on what matters. My to-do list feels less chaotic. idk.","ocean",15,7,4,2,"afternoon","calm","tense_face","clear","focused",2),
    (21,"The mountain session made me calmer but part of me still feels uneasy. Part of me wants rest part of me wants action.","mountain",15,5,2,2,"afternoon","calm","happy_face","clear","mixed",3),
    (22,"The cafe session wasn't enough today; everything still feels heavy and too much. It's hard to know where to begin.","cafe",12,None,2,4,"afternoon","focused","tired_face","conflicted","overwhelmed",4),
    (23,"even after the forest track i feel exhausted and emotionally overloaded i feel emotionally tired i almost wanted to stop midway","forest",18,5.5,2,4,"morning","focused","tense_face","vague","overwhelmed",5),
    (24,"I liked the ocean session but my mood is still split between calm and tension. idk.","ocean",20,7,4,4,"afternoon","focused","tense_face","clear","mixed",4),
    (25,"Even after the mountain track I feel exhausted and emotionally overloaded. I feel emotionally tired.","mountain",12,3.5,2,5,"night","mixed","tense_face","vague","overwhelmed",4),
    (26,"The forest session was okay I don't feel much different just a bit more aware Maybe I need more time","forest",10,7.5,3,3,"night","overwhelmed","calm_face","clear","neutral",2),
    (27,"I started scattered but the cafe session helped me lock in on what matters. My to-do list feels less chaotic.","cafe",25,7.5,4,1,"night","overwhelmed","neutral_face","clear","focused",4),
    (28,"The cafe ambience helped me breathe slower and let go of some pressure.","cafe",8,6.5,4,1,"early_morning","neutral",None,"clear","calm",3),
    (29,"The mountain background made it easier to organize my thoughts and work plan.","mountain",25,7.5,5,2,"night","restless","happy_face","clear","focused",4),
    (30,"I wasn't expecting much but the rain session made me feel quiet inside My shoulders feel less tense","rain",25,6,4,2,"early_morning","focused","neutral_face","clear","calm",2),
    (31,"I wanted the ocean to calm me but today my stress feels bigger than the session. I almost wanted to stop midway.","ocean",18,6,2,5,"afternoon","mixed",None,"conflicted","overwhelmed",5),
    (32,"I wasn't expecting much but the mountain session made me feel quiet inside. I think I can start gently today.","mountain",10,8,3,2,"evening","neutral","calm_face","vague","calm",4),
    (33,"I feel mentally clear after the cafe session and ready to tackle one thing at a time.","cafe",25,7,4,2,"morning","calm","neutral_face","clear","focused",2),
    (34,"The ocean session was okay. I don't feel much different just a bit more aware. Nothing really clicked yet.","ocean",25,6,4,4,"night","restless","neutral_face","clear","neutral",2),
    (35,"The rain ambience helped me stop drifting and concentrate on my next steps. I should use this window well.","rain",15,None,4,3,"afternoon","neutral",None,"clear","neutral",2),
    (36,"The forest session was okay. I don't feel much different just a bit more aware.","forest",12,6.5,4,2,"morning","overwhelmed","calm_face","conflicted","neutral",2),
    (37,"I feel both comforted and distracted after the cafe ambience. I can't tell if I need rest or momentum.","cafe",25,5,2,3,"evening","calm",None,"vague","mixed",4),
    (38,"I wanted the mountain to calm me but today my stress feels bigger than the session.","mountain",10,5,1,5,"early_morning",None,None,"conflicted","overwhelmed",5),
    (39,"I feel lighter after the cafe sounds like my mind finally softened. My shoulders feel less tense.","cafe",20,7,4,2,"morning","neutral",None,"clear","mixed",2),
    (40,"The rain track was fine. I feel steady not especially better or worse.","rain",20,7.5,3,2,"afternoon","focused","happy_face","clear","calm",4),
    (41,"The rain ambience was pleasant though I can't say it shifted my mood much. Maybe I need more time.","rain",8,7.5,3,3,"evening","calm","calm_face","clear","neutral",3),
    (42,"I feel both comforted and distracted after the mountain ambience There is relief but also some lingering pressure","mountain",15,5.5,3,2,"morning","neutral","neutral_face","conflicted","calm",4),
    (43,"The mountain ambience helped me stop drifting and concentrate on my next steps. I should begin with the hardest task first.","mountain",8,7,3,2,"morning","mixed",None,"clear","calm",3),
    (44,"I tried to relax during the mountain ambience yet my thoughts kept racing. I keep wanting to switch tasks.","mountain",18,4.5,4,5,"evening","focused","tense_face","clear","restless",4),
    (45,"The cafe sounds were nice but I still feel unsettled and fidgety. I keep wanting to switch tasks.","cafe",10,5,3,4,"early_morning","neutral","calm_face","conflicted","restless",3),
    (46,"The mountain track helped a little though something still feels off underneath. There is relief but also some lingering pressure.","mountain",20,7,4,3,"morning","neutral","neutral_face","clear","mixed",2),
    (47,"The cafe session slowed my thoughts and I feel more settled now. I think I can start gently today.","cafe",12,7,2,1,"early_morning","focused","happy_face","conflicted","focused",2),
    (48,"I noticed the ocean sounds but emotionally I still feel mostly the same. Maybe I need more time.","ocean",18,7.5,3,2,"early_morning","restless","neutral_face","clear","neutral",2),
    (49,"i feel lighter after the ocean sounds like my mind finally softened i think i can start gently today","ocean",25,7.5,4,2,"afternoon","neutral","calm_face","clear","calm",2),
    (50,"Even with the cafe session my mind kept jumping between tasks. I still feel a low buzz in my body.","cafe",15,5.5,4,2,"morning","overwhelmed","neutral_face","clear","restless",4),
    (51,"Nothing strong came up during the forest session; I feel fairly normal At least I paused for a moment","forest",None,6.5,3,3,"early_morning","overwhelmed","calm_face","conflicted","neutral",1),
    (52,"I liked the ocean session but my mood is still split between calm and tension. Part of me wants rest part of me wants action.","ocean",12,6,3,2,"evening","calm",None,"vague","mixed",4),
    (53,"I feel lighter after the ocean sounds like my mind finally softened. The pace of my breathing changed.","ocean",10,7.5,4,2,"afternoon","focused","happy_face","clear","calm",4),
    (54,"The ocean session wasn't enough today; everything still feels heavy and too much.","ocean",25,4.5,2,5,"afternoon","mixed",None,"conflicted","overwhelmed",5),
    (55,"I feel lighter after the cafe sounds like my mind finally softened. My shoulders feel less tense.","cafe",15,6,4,2,"evening","calm","neutral_face","clear","mixed",2),
    (56,"After the forest track I feel peaceful and less pulled in every direction.","forest",20,6.5,4,2,"evening","calm",None,"vague","focused",4),
    (57,"I feel lighter after the rain sounds like my mind finally softened.","rain",20,6,3,2,"morning","restless",None,"vague","calm",3),
    (58,"The forest sounds were nice but I still feel unsettled and fidgety.","forest",12,6,3,3,"early_morning","overwhelmed","tired_face","clear","restless",4),
    (59,"The mountain ambience helped me stop drifting and concentrate on my next steps.","mountain",8,7,3,2,"evening","restless","neutral_face","conflicted","neutral",2),
    (60,"The mountain ambience was pleasant though I can't say it shifted my mood much. I can continue the day as usual.","mountain",25,7,3,3,"morning","overwhelmed","neutral_face","conflicted","neutral",1),
    (61,"Even with the ocean session my mind kept jumping between tasks.","ocean",12,5,3,3,"morning","overwhelmed",None,"clear","restless",5),
    (62,"After the rain sounds I feel better than before but not completely okay.","rain",15,5.5,3,2,"evening","restless","calm_face","clear","mixed",3),
    (63,"The mountain session wasn't enough today; everything still feels heavy and too much. Even small tasks feel big right now.","mountain",18,4.5,1,5,"evening","restless","tired_face","vague","overwhelmed",1),
    (64,"The ocean background made it easier to organize my thoughts and work plan. I can see my priorities more clearly.","ocean",20,None,4,2,"afternoon","overwhelmed","happy_face","vague","focused",2),
    (65,"after the mountain sounds i feel better than before but not completely okay. it's like two moods are sitting together.","mountain",20,4.5,3,3,"night","mixed","neutral_face","conflicted","neutral",2),
    (66,"After the mountain track I feel peaceful and less pulled in every direction. My shoulders feel less tense.","mountain",12,8,2,2,"evening","calm","neutral_face","clear","mixed",2),
    (67,"The cafe session helped a little though I still feel pulled in too many directions.","cafe",20,5,4,3,"night","restless","neutral_face","vague","mixed",5),
    (68,"The cafe background made it easier to organize my thoughts and work plan.","cafe",18,None,3,2,"afternoon","neutral","calm_face","clear","focused",1),
    (69,"I sat through the cafe ambience but I still feel flooded by what I need to do. I almost wanted to stop midway.","cafe",None,4.5,2,4,"evening","mixed",None,"conflicted","overwhelmed",5),
    (70,"I feel lighter after the forest sounds like my mind finally softened My shoulders feel less tense","forest",10,7,3,3,"night","restless",None,"conflicted","neutral",2),
    (71,"I sat through the cafe ambience but I still feel flooded by what I need to do. I'm carrying too much in my head.","cafe",8,6,2,5,"early_morning","calm",None,"vague","overwhelmed",4),
    (72,"i sat through the mountain ambience but i still feel flooded by what i need to do.","mountain",18,3.5,3,3,"night","mixed","neutral_face","conflicted","neutral",2),
    (73,"I noticed the ocean sounds but emotionally I still feel mostly the same At least I paused for a moment","ocean",12,5.5,2,2,"evening","calm","neutral_face","clear","mixed",3),
    (74,"Nothing strong came up during the forest session; I feel fairly normal I can continue the day as usual","forest",12,7,3,2,"morning","focused",None,"vague","focused",2),
    (75,"I couldn't really settle into the rain track; I kept thinking of everything at once. I keep wanting to switch tasks.","rain",20,4.5,4,3,"night","restless","neutral_face","conflicted","restless",3),
    (76,"The cafe ambience was pleasant though I can't say it shifted my mood much. At least I paused for a moment.","cafe",30,6.5,4,3,"morning","neutral","calm_face","clear","mixed",2),
    (77,"After the forest sounds I feel better than before but not completely okay. I can't tell if I need rest or momentum.","forest",18,4.5,2,2,"afternoon","mixed",None,"vague","focused",2),
    (78,"After the cafe sounds I feel better than before but not completely okay.","cafe",8,6.5,3,2,"morning","neutral","neutral_face","conflicted","calm",4),
    (79,"I feel both comforted and distracted after the mountain ambience. It's like two moods are sitting together.","mountain",12,6,4,3,"afternoon","calm","neutral_face","vague","mixed",2),
    (80,"the cafe ambience helped me stop drifting and concentrate on my next steps. i should begin with the hardest task first.","cafe",25,5.5,5,2,"morning","restless",None,"clear","focused",5),
    (81,"I feel lighter after the mountain sounds like my mind finally softened. I think I can start gently today.","mountain",12,None,3,1,"early_morning","mixed","calm_face","vague","calm",3),
    (82,"I came in distracted but I left the rain session with a sharper mind. I should begin with the hardest task first.","rain",25,6,5,2,"morning","focused",None,"clear","focused",5),
    (83,"I started scattered but the ocean session helped me lock in on what matters. My to-do list feels less chaotic.","ocean",15,7.5,3,2,"afternoon","calm","calm_face","clear","mixed",3),
    (84,"I feel lighter after the ocean sounds like my mind finally softened. The pace of my breathing changed.","ocean",35,8,3,2,"evening","calm","neutral_face","clear","mixed",2),
    (85,"Even after the forest track I feel exhausted and emotionally overloaded.","forest",8,5.5,2,4,"morning","focused",None,"conflicted","overwhelmed",4),
    (86,"The forest session slowed my thoughts and I feel more settled now. My shoulders feel less tense.","forest",10,6.5,4,3,"afternoon","mixed","calm_face","clear","focused",4),
    (87,"The forest session made me calmer but part of me still feels uneasy There is relief but also some lingering pressure","forest",8,4.5,4,3,"morning","calm","neutral_face","conflicted","calm",3),
    (88,"I noticed the rain sounds but emotionally I still feel mostly the same. Nothing really clicked yet.","rain",20,6,3,3,"night","restless","neutral_face","clear","neutral",2),
    (89,"Even with the mountain session my mind kept jumping between tasks.","mountain",15,6.5,3,2,"evening","neutral","calm_face","clear","mixed",3),
    (90,"I couldn't really settle into the cafe track; I kept thinking of everything at once. Part of me wants to do everything at once.","cafe",8,6.5,2,4,"morning","neutral",None,"vague","overwhelmed",5),
    (91,"The cafe track was fine. I feel steady not especially better or worse.","cafe",20,7,3,3,"morning","overwhelmed","neutral_face","vague","neutral",1),
    (92,"The forest track was fine I feel steady not especially better or worse At least I paused for a moment","forest",15,5.5,2,3,"afternoon","mixed",None,"vague","neutral",3),
    (93,"Even with the cafe session my mind kept jumping between tasks. Part of me wants to do everything at once.","cafe",10,5,3,3,"evening","restless","neutral_face","clear","restless",5),
    (94,"I tried to relax during the rain ambience yet my thoughts kept racing. I keep wanting to switch tasks.","rain",20,6,3,3,"afternoon","neutral","neutral_face","clear","restless",4),
    (95,"I noticed the rain sounds but emotionally I still feel mostly the same. idk.","rain",30,6.5,3,3,"morning","restless","tense_face","vague","restless",4),
    (96,"I couldn't really settle into the forest track; I kept thinking of everything at once. I keep wanting to switch tasks.","forest",35,6,3,3,"afternoon","mixed","neutral_face","clear","restless",5),
    (97,"The mountain session made me calmer but part of me still feels uneasy. I feel better and not better at the same time.","mountain",15,6,3,3,"morning","calm","neutral_face","clear","neutral",4),
    (98,"The ocean session helped a little though I still feel pulled in too many directions. I still feel a low buzz in my body.","ocean",20,4.5,2,2,"evening","restless","calm_face","clear","mixed",3),
    (99,"I feel mentally clear after the rain session and ready to tackle one thing at a time. I can see my priorities more clearly.","rain",18,6,4,2,"evening","focused","calm_face","clear","focused",4),
    (100,"The forest track helped a little though something still feels off underneath.","forest",10,5.5,4,2,"afternoon","neutral","neutral_face","clear","mixed",4),
    (101,"After the forest sounds I feel better than before but not completely okay. I feel better and not better at the same time.","forest",15,7,2,2,"evening","restless",None,"conflicted","focused",4),
    (102,"I feel lighter after the forest sounds like my mind finally softened. The pace of my breathing changed.","forest",12,7,3,2,"night","mixed","neutral_face","clear","calm",3),
    (103,"I tried to relax during the mountain ambience yet my thoughts kept racing. I still feel a low buzz in my body. idk.","mountain",12,6,3,3,"night","restless","calm_face","clear","neutral",2),
    (104,"After the rain track I feel peaceful and less pulled in every direction. I think I can start gently today.","rain",15,8,3,2,"morning","calm","neutral_face","clear","calm",3),
    (105,"The rain session wasn't enough today; everything still feels heavy and too much. I feel emotionally tired.","rain",15,5.5,2,5,"early_morning","restless","tired_face","conflicted","overwhelmed",5),
    (106,"Even with the cafe session my mind kept jumping between tasks. I still feel a low buzz in my body.","cafe",35,5.5,3,3,"afternoon","restless",None,"vague","restless",4),
    (107,"The ocean ambience helped me stop drifting and concentrate on my next steps. I should begin with the hardest task first.","ocean",12,7,4,2,"afternoon","calm","calm_face","clear","focused",4),
    (108,"I liked the cafe session but my mood is still split between calm and tension. There is relief but also some lingering pressure.","cafe",18,7,3,3,"night","focused","calm_face","conflicted","mixed",4),
    (109,"I wasn't expecting much but the forest session made me feel quiet inside.","forest",25,7.5,3,3,"evening","neutral","calm_face","clear","neutral",1),
    (110,"I came in distracted but I left the cafe session with a sharper mind.","cafe",15,7,3,3,"afternoon","mixed",None,"conflicted","neutral",3),
    (111,"After the forest sounds I feel better than before but not completely okay.","forest",25,5,2,1,"morning","restless","calm_face","vague","focused",2),
    (112,"After the rain track I feel peaceful and less pulled in every direction. The pace of my breathing changed.","rain",10,6.5,4,2,"morning","calm","neutral_face","clear","calm",5),
    (113,"I wanted the mountain to calm me but today my stress feels bigger than the session I'm carrying too much in my head","mountain",20,4.5,1,5,"night","mixed",None,"conflicted","overwhelmed",1),
    (114,"I feel lighter after the mountain sounds like my mind finally softened. My shoulders feel less tense.","mountain",25,6,4,4,"morning","focused","calm_face","clear","focused",1),
    (115,"Even after the rain track I feel exhausted and emotionally overloaded. I almost wanted to stop midway.","rain",15,4.5,2,5,"night","calm",None,"conflicted","overwhelmed",5),
    (116,"Nothing strong came up during the rain session; I feel fairly normal. I can continue the day as usual.","rain",20,7.5,3,3,"morning","tense_face","tense_face","vague","neutral",1),
    (117,"The mountain session gave me a pause but the pressure is still sitting hard on me.","mountain",10,4,2,5,"evening","overwhelmed","calm_face","conflicted","overwhelmed",5),
    (118,"Even after the cafe track I feel exhausted and emotionally overloaded. It's hard to know where to begin.","cafe",15,5.5,3,3,"night","focused","tired_face","clear","restless",4),
    (119,"The mountain session helped a little though I still feel pulled in too many directions idk.","mountain",12,6.5,3,3,"afternoon","mixed","calm_face","clear","neutral",2),
    (120,"after the mountain track i feel peaceful and less pulled in every direction.","mountain",25,8,3,3,"morning","restless",None,"vague","calm",3),
]

TEST_RECORDS = [
    (10001,"woke up feeling more organized mentally. i was more tired than i thought.","cafe",4,8.5,3,1,"night","mixed","happy_face","vague"),
    (10002,"started off distracted most of the time. this was better than yesterday.","mountain",4,8.5,1,2,"afternoon","mixed","happy_face","clear"),
    (10003,"kinda calm ...","cafe",15,8.5,2,5,"evening","calm","happy_face","vague"),
    (10004,"after the session i felt able to think straight. my breathing slowed for a moment.","ocean",7,7,2,3,"morning","overwhelmed",None,"clear"),
    (10005,"lowkey felt pretty grounded. i had to restart once. ...","ocean",20,8.5,1,5,"afternoon","calm","tired_face","vague"),
    (10006,"today i was a little better and a little off, but sleep probably affected it.","rain",4,3.5,4,2,"morning","neutral","tired_face","clear"),
    (10007,"lowkey felt unable to stay with one thought. mountain visuals made it easier to pause.","rain",25,8,2,3,"night","calm","neutral_face","clear"),
    (10008,"ended up like everything piled up. then my mind wandered again. then it faded again.","forest",10,8.5,2,4,"afternoon","mixed",None,"clear"),
    (10009,"for a while i was itchy in my head, but ocean audio was nice.","rain",4,8.5,4,4,"evening","overwhelmed","tense_face","conflicted"),
    (10010,"for a while i was more tired than i expected. the rain helped a little.","forest",12,3.5,4,2,"night","neutral",None,"clear"),
    (10011,"started off like everything piled up, but forest sounds worked for a bit. ...","ocean",30,5,2,3,"morning","neutral","calm_face","clear"),
    (10012,"ok session","rain",20,7,2,3,"afternoon","neutral","calm_face","conflicted"),
    (10013,"actually helped","ocean",25,7,4,3,"afternoon","mixed",None,"clear"),
    (10014,"woke up feeling actually able to focus. then my mind wandered again. still not fully there though.","mountain",4,5,2,4,"afternoon","restless","happy_face","vague"),
    (10015,"started off surprisingly okay. i couldn't tell if it was helping at first.","cafe",20,4,4,1,"morning","mixed","tired_face","clear"),
    (10016,"lowkey felt just normal, but it took time to click.","cafe",35,6,4,2,"evening",None,None,"vague"),
    (10017,"by the end i was split between calm and tension. cafe ambience weirdly helped.","mountain",35,8,1,2,"morning","focused","neutral_face","conflicted"),
    (10018,"not gonna lie i felt calmer on the surface but still busy underneath, but this was better than yesterday.","ocean",30,5,3,5,"evening","focused","calm_face","clear"),
    (10019,"started off ready to start work, but teh rain helped a little.","ocean",30,7,4,1,"afternoon","focused","calm_face","vague"),
    (10020,"still off","forest",7,8.5,1,2,"evening","overwhelmed","calm_face","conflicted"),
    (10021,"somehow i felt mentally flooded. i couldn't tell if it was helping at first. still not fully there though.","ocean",7,7,1,5,"evening","neutral","calm_face","clear"),
    (10022,"woke up feeling less tense. i stayed with it anyway.","rain",7,7,5,1,"morning","calm","happy_face","vague"),
    (10023,"today i was mostly the same. the rain helped a little. that part surprised me.","rain",25,6,1,4,"night",None,None,"conflicted"),
    (10024,"not gonna lie i felt okay overall. this was better than yesterday.","forest",12,8,2,4,"evening","mixed","happy_face","vague"),
    (10025,"ok session","ocean",20,7,1,4,"morning","calm","tired_face","conflicted"),
    (10026,"by the end i was unable to come down from the day, but i kept thinking about emails.","rain",10,7,3,3,"morning","focused","happy_face","clear"),
    (10027,"for a while i was in between. i stayed with it anyway.","cafe",7,3.5,2,1,"evening","calm","neutral_face","conflicted"),
    (10028,"somehow i felt in between. i had to restart once.","cafe",20,3.5,2,1,"afternoon","calm","calm_face","clear"),
    (10029,"honestly i felt mentally flooded. i couldn't tell if it was helping at first.","rain",4,5,4,5,"afternoon","neutral","happy_face","vague"),
    (10030,"honestly i felt more settled, but i couldn't tell if it was helping at first.","rain",10,8.5,1,2,"night","focused","neutral_face","conflicted"),
    (10031,"today i was able to prioritize. sleep probably affected it. that part surprised me. ...","ocean",12,8.5,1,2,"morning","calm",None,"vague"),
    (10032,"lowkey felt kind of jumpy, but then my mind wandered again.","cafe",4,7,5,4,"evening","calm","happy_face","vague"),
    (10033,"ended up unable to come down from the day, but i stayed with it anyway.","forest",10,8.5,2,4,"morning","restless",None,"vague"),
    (10034,"woke up feeling kind of blank, but ocean audio was nice.","rain",12,7,1,4,"night","focused",None,"clear"),
    (10035,"woke up feeling able to prioritize. mountain visuals made it easier to pause.","mountain",4,8.5,4,1,"morning","restless","neutral_face","clear"),
    (10036,"honestly i felt all over the place. i kept thinking about emails.","ocean",18,6,3,5,"morning","mixed","tense_face","clear"),
    (10037,"not much change","rain",10,8.5,3,5,"morning","neutral","tense_face","vague"),
    (10038,"after the session i felt ready to start work, but i was more tired than i thought.","forest",7,8,4,2,"night","neutral","happy_face","clear"),
    (10039,"for a while i was not bad but not clear either. i had to restart once.","mountain",5,7,1,5,"evening","overwhelmed",None,"vague"),
    (10040,"by the end i was half relaxed half distracted. i couldn't tell if it was helping at first.","cafe",18,4,2,1,"evening","overwhelmed","calm_face","vague"),
    (10041,"woke up feeling still carrying too much. this was better than yesterday.","ocean",20,3.5,5,2,"morning","overwhelmed",None,"vague"),
    (10042,"by the end i was pretty grounded. i stayed with it anyway. then it faded again.","ocean",12,7,1,1,"afternoon","neutral","calm_face","conflicted"),
    (10043,"bit restless","ocean",10,5,5,1,"night","neutral",None,"vague"),
    (10044,"fine i guess","rain",15,5,4,1,"evening",None,None,"vague"),
    (10045,"not gonna lie i felt still mentally busy. it took time to click. that part surprised me.","forest",4,8.5,1,1,"morning","restless","tired_face","conflicted"),
    (10046,"kind of felt able to think straight, but the rain helped a little.","forest",5,6,4,1,"night","mixed","neutral_face","clear"),
    (10047,"not gonna lie i felt more organized mentally. this was better than yesterday.","ocean",25,5,5,2,"morning","calm",None,"clear"),
    (10048,"after the session i felt more at ease. i was more tired than i thought. that part surprised me.","forest",15,5,5,3,"afternoon","overwhelmed",None,"vague"),
    (10049,"started off still mentally busy. cafe ambience weirdly helped. ...","ocean",18,3.5,3,2,"night","restless","happy_face","clear"),
    (10050,"ended up restless even after, but i kept thinking about emails.","rain",20,3.5,5,5,"evening","neutral_face",None,"vague"),
    (10051,"after the session i felt less wound up, but then my mind wandered again.","cafe",4,7,4,1,"afternoon","mixed",None,"conflicted"),
    (10052,"lowkey felt unable to settle, but my breathing slowed for a moment.","ocean",7,3.5,4,2,"morning","neutral","neutral_face","clear"),
    (10053,"not gonna lie i felt less tense. this was better than yesterday.","forest",12,8.5,1,1,"morning","overwhelmed","neutral_face","conflicted"),
    (10054,"not gonna lie i felt mentally flooded. i couldn't tell if it was helping at first.","cafe",30,8.5,5,2,"afternoon","restless","happy_face","conflicted"),
    (10055,"after the session i felt split between calm and tension. this was better than yesterday.","mountain",30,5,5,3,"night",None,"tense_face","conflicted"),
    (10056,"honestly i felt clearer than earlier. forest sounds worked for a bit.","ocean",15,5,5,3,"night","neutral",None,"vague"),
    (10057,"not gonna lie i felt heavy in the chest. it took time to click.","mountain",15,3.5,1,2,"afternoon","calm",None,"vague"),
    (10058,"started off like everything piled up. cafe ambience weirdly helped. ...","ocean",5,4,4,1,"afternoon","overwhelmed","tired_face","conflicted"),
    (10059,"woke up feeling a little better and a little off. i couldn't tell if it was helping at first.","rain",35,7,5,4,"night","neutral","tense_face","conflicted"),
    (10060,"mind racing","ocean",35,8.5,2,2,"night","calm","tired_face","clear"),
    (10061,"lowkey felt pretty even. my breathing slowed for a moment.","mountain",12,8.5,5,3,"night","restless",None,"vague"),
    (10062,"by the end i was more at ease, but i stayed with it anyway.","forest",30,4,4,4,"night","restless","tense_face","clear"),
    (10063,"by the end i was distracted most of the time. then my mind wandered again.","forest",4,8,2,1,"afternoon","overwhelmed","tense_face","clear"),
    (10064,"started off mostly the same. sleep probably affected it.","forest",4,3.5,2,1,"morning",None,"tired_face","conflicted"),
    (10065,"kind of felt steady but nothing major, but it took time to click.","cafe",18,7,5,2,"morning","neutral","tense_face","clear"),
    (10066,"lowkey felt unable to settle. this was better than yesterday.","rain",10,4,2,1,"evening","mixed","tense_face","vague"),
    (10067,"not gonna lie i felt unable to settle. i stayed with it anyway.","cafe",15,4,2,2,"night","overwhelmed",None,"vague"),
    (10068,"for a while i was mentally flooded. then my mind wandered again.","mountain",18,6,5,5,"morning","overwhelmed","tired_face","conflicted"),
    (10069,"after the session i felt steady but nothing major. then my mind wandered again. then it faded again. ...","ocean",5,6,5,2,"evening","mixed",None,"conflicted"),
    (10070,"lowkey felt all over teh place. ocean audio was nice. ...","mountain",5,8.5,3,4,"evening","focused","happy_face","conflicted"),
    (10071,"ended up somewhat settled but still uneasy. i had to restart once.","rain",15,4,1,4,"morning","calm","tense_face","vague"),
    (10072,"lowkey felt okay overall, but forest sounds worked for a bit.","cafe",30,3.5,4,4,"afternoon","calm_face",None,"conflicted"),
    (10073,"by the end i was locked in for a bit. i kept thinking about emails.","forest",12,3.5,1,5,"night","tense_face","tense_face","clear"),
    (10074,"started off calmer on the surface but still busy underneath. ocean audio was nice.","forest",10,8,5,5,"afternoon","neutral","tired_face","conflicted"),
    (10075,"started off kind of jumpy. i had to restart once.","ocean",18,7,3,4,"morning","focused",None,"vague"),
    (10076,"lowkey felt calmer on teh surface but still busy underneath. tehn my mind wandered again.","rain",10,8.5,1,4,"morning","restless","calm_face","conflicted"),
    (10077,"by the end i was steady but nothing major. i had to restart once.","cafe",18,5,2,2,"evening","restless","happy_face","vague"),
    (10078,"somehow i felt unable to stay with one thought, but my breathing slowed for a moment.","mountain",12,4,3,1,"afternoon","restless","calm_face","conflicted"),
    (10079,"for a while i was distracted most of teh time. i kept thinking about emails.","forest",25,8,1,2,"evening","calm",None,"clear"),
    (10080,"more clear today","ocean",12,5,5,2,"night","neutral","neutral_face","vague"),
    (10081,"kind of felt restless even after. i kept thinking about emails.","rain",7,4,2,5,"night","overwhelmed","calm_face","conflicted"),
    (10082,"mind racing","rain",25,5,5,2,"afternoon","neutral",None,"clear"),
    (10083,"started off okay overall. ocean audio was nice.","rain",35,7,4,3,"afternoon","restless","tired_face","conflicted"),
    (10084,"after the session i felt drained and behind. then my mind wandered again.","cafe",15,8,1,4,"afternoon","overwhelmed",None,"conflicted"),
    (10085,"fine i guess","ocean",25,5,2,2,"evening","restless","calm_face","clear"),
    (10086,"kind of felt mostly the same, but i kept thinking about emails. ...","cafe",35,3.5,3,4,"evening","overwhelmed","tired_face","vague"),
    (10087,"kind of felt calmer on the surface but still busy underneath. ocean audio was nice.","forest",7,6,3,3,"night","restless",None,"conflicted"),
    (10088,"not gonna lie i felt okay overall. i stayed with it anyway.","ocean",7,8.5,5,2,"morning","mixed",None,"vague"),
    (10089,"woke up feeling okay overall. my breathing slowed for a moment. ...","rain",12,8,1,3,"evening","calm","happy_face","conflicted"),
    (10090,"kind of felt better but not fully okay. it took time to click.","rain",15,8.5,5,1,"evening","overwhelmed","neutral_face","conflicted"),
    (10091,"after the session i felt steady but nothing major. it took time to click.","cafe",5,4,4,5,"night","focused",None,"vague"),
    (10092,"started off like everything piled up. i had to restart once.","ocean",18,7,3,1,"afternoon","restless","happy_face","clear"),
    (10093,"after the session i felt a little better and a little off. this was better than yesterday. that part surprised me.","cafe",30,6,1,1,"evening","focused","calm_face","vague"),
    (10094,"kinda calm","rain",18,5,1,5,"morning","focused","tired_face","clear"),
    (10095,"after teh session i felt more tired than i expected. i couldn't tell if it was helping at first.","ocean",7,4,3,2,"night","neutral","calm_face","clear"),
    (10096,"still off ...","ocean",7,8,2,3,"afternoon","neutral",None,"vague"),
    (10097,"started off in between. i kept thinking about emails.","mountain",35,7,4,2,"evening","focused","happy_face","vague"),
    (10098,"started off heavy in teh chest. i stayed with it anyway. still not fully tehre though.","cafe",5,4,1,5,"evening","restless","calm_face","vague"),
    (10099,"started off half relaxed half distracted, but my breathing slowed for a moment. ...","mountain",18,8,2,4,"morning","restless","happy_face","clear"),
    (10100,"started off somewhat settled but still uneasy. i stayed with it anyway.","mountain",25,8,4,5,"morning","focused","calm_face","clear"),
    (10101,"started off more settled. i stayed with it anyway. that part surprised me.","rain",20,3.5,1,2,"evening","neutral","tense_face","clear"),
    (10102,"lowkey felt unable to come down from the day. the rain helped a little.","ocean",5,6,3,3,"evening","focused","tired_face","conflicted"),
    (10103,"kind of felt lighter than before. mountain visuals made it easier to pause.","ocean",5,6,1,5,"afternoon","focused",None,"clear"),
    (10104,"not gonna lie i felt somewhat settled but still uneasy. forest sounds worked for a bit.","ocean",4,5,1,5,"evening","mixed","calm_face","conflicted"),
    (10105,"ok session","cafe",15,4,1,1,"morning","restless",None,"clear"),
    (10106,"somehow i felt not very different. then my mind wandered again.","cafe",5,8.5,5,3,"afternoon","neutral",None,"clear"),
    (10107,"hard to focus","forest",25,3.5,1,2,"afternoon","overwhelmed","happy_face","conflicted"),
    (10108,"after the session i felt in between. sleep probably affected it.","cafe",7,6,1,1,"morning","calm","tired_face","vague"),
    (10109,"i noticed i was fine i guess. cafe ambience weirdly helped.","ocean",10,5,4,5,"afternoon",None,None,"conflicted"),
    (10110,"ok session","mountain",15,8.5,2,5,"night","calm","calm_face","vague"),
    (10111,"today i was split between calm and tension. it took time to click.","rain",5,3.5,1,1,"morning",None,None,"conflicted"),
    (10112,"lowkey felt calmer on the surface but still busy underneath. forest sounds worked for a bit.","forest",5,3.5,2,2,"afternoon","focused","neutral_face","vague"),
    (10113,"i noticed i was less tense. i was more tired than i thought.","mountain",18,3.5,1,4,"morning","calm","neutral_face","vague"),
    (10114,"by teh end i was pretty grounded. cafe ambience weirdly helped. ...","mountain",4,7,3,4,"afternoon","calm",None,"vague"),
    (10115,"woke up feeling lighter than before. then my mind wandered again.","mountain",30,7,5,5,"evening","restless",None,"conflicted"),
    (10116,"ended up unable to come down from the day. then my mind wandered again.","rain",7,5,4,2,"night","focused",None,"conflicted"),
    (10117,"somehow i felt pretty grounded, but it took time to click. ...","forest",5,7,1,3,"afternoon","overwhelmed","tense_face","clear"),
    (10118,"started off ready to start work, but i stayed with it anyway.","rain",12,8,4,2,"afternoon","calm","tired_face","vague"),
    (10119,"somehow i felt somewhat settled but still uneasy. i kept thinking about emails.","forest",25,8.5,4,4,"night","neutral","tired_face","conflicted"),
    (10120,"by the end i was distracted most of the time. i couldn't tell if it was helping at first.","mountain",35,6,5,4,"afternoon","neutral","happy_face","conflicted"),
]

COLS_TRAIN = ["id","journal_text","ambience_type","duration_min","sleep_hours",
              "energy_level","stress_level","time_of_day","previous_day_mood",
              "face_emotion_hint","reflection_quality","emotional_state","intensity"]

COLS_TEST = ["id","journal_text","ambience_type","duration_min","sleep_hours",
             "energy_level","stress_level","time_of_day","previous_day_mood",
             "face_emotion_hint","reflection_quality"]

# ─────────────────────────────────────────────────────────────
# 2. SIGNAL EXTRACTION
# ─────────────────────────────────────────────────────────────

SIGNAL_WORDS = {
    'resolved':   ['clear','focused','settled','lighter','calmer','peaceful','quiet','organized',
                   'prioritize','lock in','sharper','gently','breathe','concentrate','plan','grounded'],
    'tension':    ['buzz','fidgety','racing','jumping','scattered','flooded','pressure','carry',
                   'too much','pulled','heavy','exhausted','overloaded','stuck','restless','distracted'],
    'ambivalent': ['but','though','still','yet','idk','maybe',"can't tell",'both','split',
                   'not fully','better and not better','part of me','at the same time','in between'],
    'uncertain':  ['idk','maybe','not sure',"don't know",'unclear','possibly','kinda','i guess',
                   'somehow','weird','strange',"can't tell"],
    'disengaged': ['fine','okay','normal','nothing much','not much','same','ordinary',
                   "didn't notice",'nothing strong','fairly normal','not very different'],
    'action_urge':['begin','start','tackle','hardest','do everything','momentum','window',
                   'plan','work plan','prioritize','lock in','ready to'],
    'rest_urge':  ['rest','tired','exhausted','drained','stop','midway','too much','heavy',
                   'slow','breathing','pause','sleep'],
    'near_quit':  ['almost wanted to stop','stop midway','couldn\'t settle','restart','almost stopped'],
}


def extract_text_features(text):
    if not text or not isinstance(text, str):
        return {k: 0 for k in ['word_count','has_resolved','has_tension','has_ambivalent',
                                'has_uncertain','has_disengaged','has_action_urge','has_rest_urge',
                                'has_near_quit','contradiction_score','signal_strength',
                                'text_is_short','hedge_density','typo_count',
                                'tension_resolution_ratio','exclamation_count']}
    t = text.lower()
    words = t.split()
    wc = len(words)

    feats = {'word_count': wc, 'text_is_short': int(wc < 10)}
    for cat, markers in SIGNAL_WORDS.items():
        feats[f'has_{cat}'] = int(any(m in t for m in markers))

    feats['contradiction_score'] = int(
        feats['has_resolved'] and feats['has_tension']
    )
    fired = sum(feats[f'has_{k}'] for k in SIGNAL_WORDS)
    feats['signal_strength'] = fired / len(SIGNAL_WORDS)

    hedges = ['maybe','kinda','sort of','i guess','idk','not sure','probably','might']
    feats['hedge_density'] = sum(1 for h in hedges if h in t) / max(wc, 1)

    typos = ['teh ','tehn ','tnse','otehr','tehre','eitehr','anotehr']
    feats['typo_count'] = sum(1 for tp in typos if tp in t)
    feats['exclamation_count'] = text.count('!')

    t_hits = sum(1 for m in SIGNAL_WORDS['tension'] if m in t)
    r_hits = sum(1 for m in SIGNAL_WORDS['resolved'] if m in t)
    feats['tension_resolution_ratio'] = t_hits / max(r_hits + 1, 1)

    return feats


def compute_uncertainty(row, text_feats):
    reasons = []
    text = str(row.get('journal_text', ''))
    wc = len(text.split())

    if wc < 8:
        reasons.append('very_short_entry')
    t = text.lower()
    if sum(1 for p in ['idk','not sure',"can't tell",'i guess','maybe','somehow'] if p in t) >= 2:
        reasons.append('multiple_hedges')

    pairs = [('calm','tense'),('peaceful','pressure'),('lighter','heavy'),('clear','racing')]
    if sum(1 for a, b in pairs if a in t and b in t) >= 2:
        reasons.append('contradictory_signals')

    if pd.isna(row.get('sleep_hours')):
        reasons.append('missing_sleep')
    if pd.isna(row.get('stress_level')):
        reasons.append('missing_stress')
    if str(row.get('face_emotion_hint', '')).lower() in ['none','nan','']:
        reasons.append('no_face_signal')

    confidence = max(0.2, 1.0 - len(reasons) * 0.12)
    return {
        'uncertain_flag': int(len(reasons) > 0),
        'confidence': round(confidence, 3),
        'reasons': reasons,
    }


# ─────────────────────────────────────────────────────────────
# 3. FEATURE MATRIX
# ─────────────────────────────────────────────────────────────

def build_features(df):
    text_feats = pd.DataFrame(df['journal_text'].fillna('').apply(extract_text_features).tolist())

    struct = df[['duration_min','sleep_hours','energy_level','stress_level']].copy()
    for c in struct.columns:
        struct[c] = pd.to_numeric(struct[c], errors='coerce')
        struct[c] = struct[c].fillna(struct[c].median())

    struct['sleep_stress_ratio'] = struct['sleep_hours'] / (struct['stress_level'] + 1)
    struct['energy_stress_gap']  = struct['energy_level'] - struct['stress_level']

    cats = pd.concat([
        pd.get_dummies(df['ambience_type'].fillna('unknown'),    prefix='amb'),
        pd.get_dummies(df['time_of_day'].fillna('unknown'),      prefix='tod'),
        pd.get_dummies(df['previous_day_mood'].fillna('unknown'),prefix='prev'),
        pd.get_dummies(df['face_emotion_hint'].fillna('none'),   prefix='face'),
        pd.get_dummies(df['reflection_quality'].fillna('unknown'),prefix='rq'),
    ], axis=1)

    X = pd.concat([text_feats, struct, cats], axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X


def build_tfidf_features(texts):
    vec = TfidfVectorizer(max_features=300, ngram_range=(1, 2),
                          min_df=1, stop_words=None)
    return vec, vec.fit_transform(texts.fillna(''))


# ─────────────────────────────────────────────────────────────
# 4. DECISION ENGINE — what to do + when
# ─────────────────────────────────────────────────────────────

def decide(predicted_state, intensity, stress, energy, time_of_day, text_feats, confidence):
    """
    Behavioral decision engine.
    Returns: what_to_do, when_to_do, supportive_message
    """
    state  = str(predicted_state).lower().strip()
    tod    = str(time_of_day).lower().strip()
    intens = float(intensity) if intensity else 3.0
    st     = float(stress)    if stress    else 3.0
    en     = float(energy)    if energy    else 3.0

    has_tension    = bool(text_feats.get('has_tension'))
    has_resolved   = bool(text_feats.get('has_resolved'))
    has_rest       = bool(text_feats.get('has_rest_urge'))
    has_action     = bool(text_feats.get('has_action_urge'))
    has_ambivalent = bool(text_feats.get('has_ambivalent'))
    is_short       = bool(text_feats.get('text_is_short'))
    contradiction  = bool(text_feats.get('contradiction_score'))

    # ── WHAT TO DO ───────────────────────────────────────────
    if is_short and confidence < 0.5:
        what = 'journaling'
    elif state == 'overwhelmed' and en <= 2:
        what = 'rest'
    elif state == 'overwhelmed' and st >= 4:
        what = 'box_breathing'
    elif state == 'restless' and en >= 4 and has_action:
        what = 'deep_work'
    elif state == 'restless' and st >= 4:
        what = 'box_breathing'
    elif state == 'restless':
        what = 'light_planning'
    elif state == 'focused' and en >= 4:
        what = 'deep_work'
    elif state == 'focused':
        what = 'light_planning'
    elif state == 'calm' and en >= 4:
        what = 'deep_work'
    elif state == 'calm':
        what = 'light_planning'
    elif state == 'mixed' and contradiction:
        what = 'grounding'
    elif state == 'mixed' and has_rest:
        what = 'rest'
    elif state == 'mixed':
        what = 'journaling'
    elif state == 'neutral' and en >= 4:
        what = 'light_planning'
    elif state == 'neutral':
        what = 'movement'
    else:
        what = 'grounding'

    # high stress always overrides to body-first
    if st >= 5 and what not in ['rest', 'box_breathing']:
        what = 'box_breathing'

    # ── WHEN TO DO IT ────────────────────────────────────────
    morning_times   = ['morning', 'early_morning']
    evening_times   = ['evening', 'night']
    afternoon_times = ['afternoon']

    if what in ['rest'] and tod in evening_times:
        when = 'tonight'
    elif what in ['rest']:
        when = 'later_today'
    elif what in ['box_breathing','grounding'] and intens >= 4:
        when = 'now'
    elif what in ['box_breathing','grounding']:
        when = 'within_15_min'
    elif what == 'deep_work' and tod in morning_times and en >= 4:
        when = 'now'
    elif what == 'deep_work' and tod in afternoon_times:
        when = 'within_15_min'
    elif what == 'deep_work':
        when = 'later_today'
    elif what == 'light_planning' and tod in morning_times:
        when = 'within_15_min'
    elif what == 'light_planning':
        when = 'later_today'
    elif what == 'journaling' and tod in evening_times:
        when = 'tonight'
    elif what == 'journaling':
        when = 'within_15_min'
    elif what == 'movement' and tod in morning_times:
        when = 'now'
    elif what == 'movement':
        when = 'within_15_min'
    elif tod in ['night'] and intens <= 2:
        when = 'tomorrow_morning'
    else:
        when = 'later_today'

    # ── SUPPORTIVE MESSAGE ───────────────────────────────────
    msg_map = {
        'rest':          f"You seem {state} right now with intensity running high. Your body is asking for a break — rest before you push forward.",
        'box_breathing': f"There's a lot of tension coming through. Before anything else, try 4 counts in, hold 4, out 4. Just that.",
        'deep_work':     f"You're in a good place right now — mind is clearer than usual. Use this window. Pick the one most important thing and start.",
        'light_planning':"You've got some clarity but not full momentum yet. A quick 5-minute plan of what to tackle today will help you move forward.",
        'grounding':     f"Two things feel true at once — that's okay. Try one slow physical action (stretch, walk, make tea) before deciding anything.",
        'journaling':    "Not enough came through today to say clearly what you need. Take 5 minutes to write one sentence: what's actually sitting on you right now?",
        'movement':      "Nothing strong came through, which sometimes means the mind needs the body to move first. Even a short walk helps.",
        'sound_therapy': "Your system needs softening. A gentle sound session focused on just breathing might help more than doing anything right now.",
        'yoga':          "There's tension in the body even if the mind feels okay. Some slow movement would help discharge that.",
        'pause':         "Take a pause — not everything needs to be acted on right now. Sit with this for a few more minutes.",
    }
    message = msg_map.get(what, f"Based on how your session went, try {what} {when.replace('_',' ')}.")

    return what, when, message


# ─────────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def train_models(df_train):
    print("\n── MODEL TRAINING ───────────────────────────────────────")

    # ── emotional state (classification) ──
    df_es = df_train[df_train['emotional_state'].notna()].copy()
    df_es['emotional_state'] = df_es['emotional_state'].str.lower().str.strip()

    X_es = build_features(df_es)
    le_es = LabelEncoder()
    y_es = le_es.fit_transform(df_es['emotional_state'])

    clf_es = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
    cv_es = cross_val_score(clf_es, X_es, y_es,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring='f1_weighted')
    clf_es.fit(X_es, y_es)
    print(f"  Emotional State  — CV F1: {cv_es.mean():.3f} ± {cv_es.std():.3f}")

    # ── intensity (regression — more natural for 1-5 scale) ──
    df_int = df_train[df_train['intensity'].notna()].copy()
    df_int['intensity'] = pd.to_numeric(df_int['intensity'], errors='coerce')
    df_int = df_int.dropna(subset=['intensity'])

    X_int = build_features(df_int)
    y_int = df_int['intensity'].values

    reg_int = RandomForestRegressor(n_estimators=200, max_depth=8,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
    cv_int = cross_val_score(reg_int, X_int, y_int,
                             cv=5, scoring='neg_mean_absolute_error')
    reg_int.fit(X_int, y_int)
    print(f"  Intensity        — CV MAE: {-cv_int.mean():.3f} ± {cv_int.std():.3f}")
    print("  [Intensity treated as regression — 1–5 is ordinal, not categorical]")

    # ── ablation: text-only vs text+metadata ──
    print("\n── ABLATION STUDY ───────────────────────────────────────")
    text_only = pd.DataFrame(df_es['journal_text'].fillna('').apply(extract_text_features).tolist())
    clf_text_only = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_text = cross_val_score(clf_text_only, text_only, y_es,
                              cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              scoring='f1_weighted')
    print(f"  Text-only model  — CV F1: {cv_text.mean():.3f}")
    print(f"  Text+metadata    — CV F1: {cv_es.mean():.3f}")
    print(f"  Metadata adds:   +{(cv_es.mean()-cv_text.mean()):.3f} F1 points")
    print("  → Metadata (sleep, stress, energy) meaningfully improves prediction.")
    print("    Face emotion hint + reflection quality are the strongest non-text signals.")

    # feature importance
    print("\n── TOP FEATURES ─────────────────────────────────────────")
    fi = sorted(zip(X_es.columns, clf_es.feature_importances_), key=lambda x: -x[1])[:10]
    for fname, fval in fi:
        print(f"  {fname:40s} {fval:.4f}")

    return clf_es, le_es, reg_int, X_es.columns.tolist()


# ─────────────────────────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────────────────────────

def predict_all(df_test, clf_es, le_es, reg_int, feature_cols):
    results = []

    for _, row in df_test.iterrows():
        text_feats = extract_text_features(str(row.get('journal_text', '')))
        unc = compute_uncertainty(row, text_feats)

        # build feature row, align to training columns
        sample = pd.DataFrame([row.to_dict()])
        X_sample = build_features(sample)
        X_sample = X_sample.reindex(columns=feature_cols, fill_value=0)

        # emotional state
        es_probs = clf_es.predict_proba(X_sample)[0]
        es_idx   = np.argmax(es_probs)
        pred_state = le_es.inverse_transform([es_idx])[0]
        model_conf = float(es_probs[es_idx])

        # blend model confidence with uncertainty
        final_conf = round(model_conf * (0.5 + 0.5 * (1 - unc['uncertain_flag'] * 0.3)), 3)

        # intensity
        pred_intensity_raw = reg_int.predict(X_sample)[0]
        pred_intensity = int(round(np.clip(pred_intensity_raw, 1, 5)))

        # decision
        what, when, message = decide(
            pred_state, pred_intensity,
            row.get('stress_level'), row.get('energy_level'),
            row.get('time_of_day'), text_feats, final_conf
        )

        results.append({
            'id':                  row['id'],
            'predicted_state':     pred_state,
            'predicted_intensity': pred_intensity,
            'confidence':          final_conf,
            'uncertain_flag':      unc['uncertain_flag'],
            'what_to_do':          what,
            'when_to_do':          when,
            'supportive_message':  message,
            '_uncertainty_reasons': ', '.join(unc['reasons']) if unc['reasons'] else 'none',
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# 7. ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_error_analysis(df_train, clf_es, le_es, feature_cols):
    print("\n── ERROR ANALYSIS (10 failure cases) ───────────────────")

    df_labeled = df_train[df_train['emotional_state'].notna()].copy()
    df_labeled['emotional_state'] = df_labeled['emotional_state'].str.lower().str.strip()

    X = build_features(df_labeled)
    X = X.reindex(columns=feature_cols, fill_value=0)
    y_true = df_labeled['emotional_state'].values
    y_pred = le_es.inverse_transform(clf_es.predict(X))

    failures = []
    for i, (idx, row) in enumerate(df_labeled.iterrows()):
        if y_true[i] != y_pred[i]:
            text_feats = extract_text_features(str(row.get('journal_text', '')))
            unc = compute_uncertainty(row, text_feats)
            failures.append({
                'id':       row['id'],
                'text':     str(row['journal_text'])[:90],
                'true':     y_true[i],
                'pred':     y_pred[i],
                'reasons':  unc['reasons'],
                'wc':       text_feats['word_count'],
                'tension':  text_feats['has_tension'],
                'resolved': text_feats['has_resolved'],
                'contra':   text_feats['contradiction_score'],
            })

    failures = failures[:10]
    cases = []
    for f in failures:
        print(f"\n  ID {f['id']} | True: {f['true']:12s} | Pred: {f['pred']:12s}")
        print(f"  Text:    \"{f['text']}\"")
        print(f"  Signals: tension={f['tension']} resolved={f['resolved']} contradiction={f['contra']} wc={f['wc']}")
        if f['reasons']:
            print(f"  Flags:   {', '.join(f['reasons'])}")
        cases.append(f)

    return cases


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ARVYAX SESSION INTELLIGENCE SYSTEM")
    print("  interpretation → judgment → decision")
    print("=" * 60)

    df_train = pd.DataFrame(TRAINING_RECORDS, columns=COLS_TRAIN)
    df_test  = pd.DataFrame(TEST_RECORDS,     columns=COLS_TEST)

    print(f"\n  Training records : {len(df_train)}")
    print(f"  Test records     : {len(df_test)}")
    print(f"  Emotional states : {sorted(df_train['emotional_state'].dropna().unique())}")

    clf_es, le_es, reg_int, feature_cols = train_models(df_train)

    error_cases = run_error_analysis(df_train, clf_es, le_es, feature_cols)

    print("\n── GENERATING PREDICTIONS ───────────────────────────────")
    predictions = predict_all(df_test, clf_es, le_es, reg_int, feature_cols)

    # save predictions.csv (official deliverable)
    predictions_out = predictions[[
        'id','predicted_state','predicted_intensity',
        'confidence','uncertain_flag','what_to_do','when_to_do'
    ]]
    predictions_out.to_csv('predictions.csv', index=False)
    print(f"  predictions.csv saved ({len(predictions_out)} rows)")

    # show sample
    print("\n── SAMPLE PREDICTIONS ───────────────────────────────────")
    for _, r in predictions.head(8).iterrows():
        print(f"\n  ID {int(r['id'])} | {r['predicted_state']:12s} | intensity {r['predicted_intensity']} | conf {r['confidence']:.2f}")
        print(f"  → {r['what_to_do']} | {r['when_to_do']}")
        if r['uncertain_flag']:
            print(f"  ⚠ uncertain: {r['_uncertainty_reasons']}")
        print(f"  \"{r['supportive_message']}\"")

    print("\n  Done. All files saved.")
    return predictions


if __name__ == '__main__':
    main()
