import os
import sys
import json
from sklearn.model_selection import train_test_split

def save_data(split, filename):
    with open(filename, 'w+') as f:
        for file in split:
            f.write(file + '\n')
    print("wrote " + filename)

data_dir, dest_path_train, dest_path_test, extension = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

data_files = []
data_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames if os.path.splitext(f)[1] == extension]
print(json.dumps(data_files[:10], indent=2))
train, test = train_test_split(data_files, test_size = 0.2)
save_data(train, dest_path_train)
save_data(test, dest_path_test)
