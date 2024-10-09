Deep dive of Transformer Model using Covid Tweets
https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification

Inspired by Andrez Karpathy's https://www.youtube.com/watch?v=kCc8FmEb1nY.

I implemented all the concepts taught by Andrez in the above video but I applied it on a Covid Tweets.

I finally trained it on 20 GB RTX 4090 for 5000 Steps.This is the output that I got

Context : LRAB https://t.co/w4dJUMpdqz The #US is already in a recession, dozens of economists say. That job loss will drag on spending, a major driver of the US economy. Consumer spending accounts for roughly 70% of economic growth. #coronavirus https://t.co


Model Output : LRAB https://t.co/w4dJUMpdqz The #US is already in a recession, dozens of economists say. That job loss will drag on spending, a major driver of the US economy. Consumer spending accounts for roughly 70% of economic growth. #coronavirus https://t.co/FitWDU00Ab?
s Hot have encourage in the city luck a minute at idiots
#Coronavirus
#coronavirusInnormalockdown #CoronavirusOutandemic #CoronavirusLockdownbroughy @asdal #coronaVirus #coronavirus weve I can you have to $0, housingconsumers to know a friend delivery it are infl crunched reform Ill. Make that aim
 -&gt;&


Number of parameters in model = 123 M
Memory taken by the parameters = 1 gb

Trying to understand ROPE using this repo
https://github.com/adalkiran/llama-nuts-and-bolts/blob/main/docs/10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md

