"""
Check that splits are patient-level disjoint and print basic stats.
"""
import os
import re

split_dir = os.path.join(os.path.dirname(__file__), '..', 'splits')
train_f = os.path.join(split_dir, 'train_list.txt')
val_f = os.path.join(split_dir, 'val_list.txt')
test_f = os.path.join(split_dir, 'test_list.txt')

S_PAT = re.compile(r'^(Case_\d+)')

def load_ids(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

train = load_ids(train_f)
val = load_ids(val_f)
test = load_ids(test_f)

def patients(samples):
    s = set()
    for ss in samples:
        m = S_PAT.match(ss)
        if m:
            s.add(m.group(1))
    return s

p_train = patients(train)
p_val = patients(val)
p_test = patients(test)

print('Counts -> samples: train=%d val=%d test=%d' % (len(train), len(val), len(test)))
print('Counts -> patients: train=%d val=%d test=%d' % (len(p_train), len(p_val), len(p_test)))

print('Intersections:')
print('train & val:', sorted(list(p_train & p_val))[:10])
print('train & test:', sorted(list(p_train & p_test))[:10])
print('val & test:', sorted(list(p_val & p_test))[:10])

if (p_train & p_val) or (p_train & p_test) or (p_val & p_test):
    print('\nERROR: patient overlap found between splits')
else:
    print('\nOK: No patient overlap between splits')

