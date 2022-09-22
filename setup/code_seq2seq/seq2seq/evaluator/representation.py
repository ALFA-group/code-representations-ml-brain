from __future__ import print_function, division

import torch
import torchtext

import code_seq2seq.seq2seq as seq2seq
from code_seq2seq.seq2seq.util.concat import torch_concat

class Representation(object):
    """ Class to gather intermediate representation produced by the model for the given dataset.

    Args:
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, device, batch_size=64):
        self.device = device
        self.batch_size = batch_size

    def get_representation(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        if self.device != 'cpu':
            model.cuda()
        
        model.eval()
        
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=self.device, train=False)
        
        with torch.no_grad():
            all_hidden = None
            for batch in batch_iterator:
                input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
                _, (encoder_output, encoder_hidden) = model(device=self.device, input_variable=input_variables, 
                                                            input_lengths=input_lengths.tolist(), 
                                                            target_variable=None)
                encoder_hidden = torch.sum(encoder_hidden, 0)
                all_hidden = torch_concat(all_hidden, encoder_hidden)

        return all_hidden
