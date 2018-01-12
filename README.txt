
To run the code:

For logistic regression:

python main.py -e data/twitter_dev_test.ner data/twitter_dev.ner data/twitter_test.ner

For CRF:

python -u main.py -T crf -e data/twitter_dev.ner data/twitter_dev_test.ner data/twitter_test.ner

(specify crf as tagger)

For word clusters:

I’ve used word clusters constituting English tweets produced by an unsupervised HMM(Brown Clustering Implementation)

[Initially, tried using Google News but it was taking quite some time to even cluster after training and loading hence switched to CMU’s twitter word clusters]

I used the largest set constituting 847,372,038 tokens.
File: 50mpaths2.txt





 