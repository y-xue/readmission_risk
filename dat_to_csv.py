import pandas as pd

def dat_to_csv(folder, cols, glen):
    bug_l = {}
    for i in range(5):
        data = pd.read_csv('%s/%s__IMPDATA%d.dat' % (folder, folder, i+1), sep='\s+', header=None, skiprows=1)
        data.columns = cols
        (n,m) = data.shape
        for j in range(n-glen,n):
            for col in cols:
                if data[col][j] == '.':
                    if col in bug_l:
                        bug_l[col] += 1
                    else:
                        bug_l[col] = 0
                    data[col][j] = 0
        data.to_csv('%s/%s_%d.csv' % (folder, folder, i), index=False)
    print 'bug_l:', bug_l