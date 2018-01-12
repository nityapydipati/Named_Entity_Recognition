import numpy as np


def back_propagate_max_prob(L,N,y,trellis,end_scores,back_pointers):
    max_curr= -9999999
    curr_pos = 0
    for i in xrange(L):
        val = trellis[N-1][i] + end_scores[i]
        if max_curr<val:
            max_curr= val
            curr_pos = i
    
    y.append(curr_pos)
    for i in xrange(N-1, 0, -1):
        y.append(back_pointers[i][curr_pos])
        curr_pos = back_pointers[i][curr_pos]

    return y,max_curr



def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []

    trellis = np.array(np.zeros(emission_scores.shape))
    trellis.fill(-9999999)
    back_pointers = np.zeros(emission_scores.shape, dtype=int)

    for i in xrange(L):
        trellis[0][i] = start_scores[i] + emission_scores[0][i]
        back_pointers[0][i] = i

    for i in xrange(N-1):
        temp = -1
        for t in xrange(L):
            for t1 in xrange(L):
                temp = emission_scores[i+1][t] + trellis[i][t1] + trans_scores[t1][t]
                if temp > trellis[i+1][t]:
                    trellis[i+1][t] = temp
                    back_pointers[i+1][t] = t1

    y,max_curr=back_propagate_max_prob(L,N,y,trellis,end_scores,back_pointers)

    #for i in xrange(N):
        # stupid sequence
        #y.append(i % L)
    # score set to 0
    y.reverse()
    return (max_curr,y)
