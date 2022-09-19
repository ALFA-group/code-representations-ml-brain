import sys
import os
import random
import torch
from torchtext.data import Dataset, Example
import pickle as pkl
from code_seq2seq.seq2seq.evaluator import Representation
from code_seq2seq.tokenize import transform_data
from code_seq2seq.train import params
from code_seq2seq.seq2seq.dataset import SourceField
import warnings
warnings.filterwarnings('ignore')

def dump_data(pth, fname, ds):
    with open(os.path.join(pth, fname), 'wb') as fp:
        torch.save(ds, fp)


class TabularDataset_From_List(Dataset):
    def __init__(self, input_list, fields, **kwargs):
        fields_dict = {}
        for (n, f) in fields:
            fields_dict[n] = (n, f)
        examples = [Example.fromdict(item, fields_dict) for item in input_list]
        super(TabularDataset_From_List, self).__init__(examples, fields, **kwargs)


def get_representation(model, tokenized_program, max_len, input_vocab, device, debug=False):
    model.to(device)
    tokenized_program = tokenized_program[:max_len]
    src = SourceField()
    dataset = TabularDataset_From_List(
        [{'src': tokenized_program}],
        [('src', src)]
    )
    src.build_vocab(dataset)    
    src.vocab = input_vocab # Overwrite vocab once `vocab` attribute has been set.
    rep = Representation(device)
    all_reps = rep.get_representation(model, dataset)
    if debug:
        print("# of unique program tokens: {}".format(len(set(tokenized_program))))
        print("Vocab length: {}".format(len(src.vocab)))
        print("Rep shape: {}".format(all_reps.shape))
        print('---')
    return all_reps


if __name__ == '__main__':
    saved_model_path, saved_vocab_path, rep_dump_path = sys.argv[1], sys.argv[2], sys.argv[3]
    data_files_path = sys.argv[4]

    if torch.cuda.is_available():
        device_count = torch.torch.cuda.device_count()
        if device_count > 0:
            device_id = random.randrange(device_count)
            device = torch.device('cuda:'+str(device_id))
            torch.cuda.set_device(device)
    else:
        device = 'cpu'

    with open(saved_model_path, 'rb') as fp:
        seq2seq_model = torch.load(fp, map_location=device)

    with open(saved_vocab_path, 'rb') as fp:
        input_vocab = pkl.load(fp)

    max_len = params['max_len']
    train_dataset = transform_data(data_files_path, debug=True)

    print('Getting representations..')
    for p in train_dataset[:5]:
        r = get_representation(seq2seq_model, p, max_len, input_vocab, device)
        dump_data(rep_dump_path, 'data_reps.torch', r)

    print('Done dumping to {}'.format(rep_dump_path))
