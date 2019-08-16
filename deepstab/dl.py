

def get_hot_coded_seq(sequence):
    """Convert a 4 base letter sequence to 4-row x-cols hot coded sequence"""
    hotsequence = np.zeros((len(sequence),4))
    for i in range(len(sequence)):
        if sequence[i] == 'A':
            hotsequence[i,0] = 1
        elif sequence[i] == 'C':
            hotsequence[i,1] = 1
        elif sequence[i] == 'G':
            hotsequence[i,2] = 1
        elif sequence[i] == 'T':
            hotsequence[i,3] = 1
        elif sequence[i] == 'N':
            hotsequence[i,0] = 0.25
            hotsequence[i,1] = 0.25
            hotsequence[i,2] = 0.25
            hotsequence[i,3] = 0.25
        elif sequence[i] == 'P':
            pass
    return hotsequence

