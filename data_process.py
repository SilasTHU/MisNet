import csv
from transformers import AutoTokenizer
import os


# parent class for dataset processor
class Data_Processor:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.DATA.plm)
        self.data_dir = args.DATA.data_dir
        self.sep_puncs = args.DATA.sep_puncs
        self.max_left_len = args.DATA.max_left_len  # the previously set max length of the left input
        self.max_right_len = args.DATA.max_right_len
        self.use_pos = args.DATA.use_pos
        self.use_eg_sent = args.DATA.use_eg_sent

    def __str__(self):
        pattern = '''Data Configs: 
        data_dir: {} 
        sep_puncs: {} 
        max_left_len: {} 
        max_right_len: {} 
        use_pos: {}
        use_eg_sent: {}'''
        return pattern.format(self.data_dir, self.sep_puncs, self.max_left_len,
                              self.max_right_len, self.use_pos, self.use_eg_sent)

    def _get_examples(self, file_dir):
        """

        :param file_dir: a vua csv file
        :return:
        """
        examples = []

        # punctuations to mark a context
        sep_puncs = [self.tokenizer.encode(sep_punc, add_special_tokens=False)[0] for sep_punc in self.sep_puncs]

        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                # example sentence may be empty, caution for processing
                sentence, label, target_position, target_word, pos_tag, gloss, eg_sent = line
                label = int(label)
                target_position = int(target_position)
                sep_id = self.tokenizer.encode(self.tokenizer.sep_token, add_special_tokens=False)[0]

                # segment ids: 1==>target word, 2==>local context, 3==>pos, 0==>others
                """
                for the left part
                """
                # convert sentence to ids: <s> sentence </s>
                ids_l = self.tokenizer.encode(sentence)
                segs_l = [0] * len(ids_l)

                # target word may be cut into word pieces by tokenizer, find the range of pieces
                tar_start, tar_end = target_align(target_position, sentence, tokenizer=self.tokenizer)
                local_start, local_end = get_local(ids_l, tar_start, sep_puncs)

                # set local segments and target segments
                segs_l[local_start: local_end] = [2] * (local_end - local_start)
                segs_l[tar_start: tar_end] = [1] * (tar_end - tar_start)

                """
                for the right part: <s> target_words </s> </s> POS </s> </s> basic_usage </s>
                """
                ids_r = self.tokenizer.encode(target_word)  # <s> target_word </s>
                segs_r = [0] * len(ids_r)
                segs_r[1:-1] = [1] * (len(ids_r) - 2)  # except <s> and </s>, rest tokens are tagged as target_word

                if self.use_pos:
                    # </s> POS </S>
                    pos_ids = [sep_id] + self.tokenizer.encode(pos_tag, add_special_tokens=False) + [sep_id]
                    pos_segs = [0] + [3] * (len(pos_ids) - 2) + [0]
                    ids_r = ids_r + pos_ids
                    segs_r = segs_r + pos_segs

                if self.use_eg_sent and eg_sent != '':
                    # </s> basic_usage </s>
                    eg_ids = [sep_id] + self.tokenizer.encode(eg_sent, add_special_tokens=False) + [sep_id]
                    eg_segs = [0] + [2] * (len(eg_ids) - 2) + [0]
                    ids_r = ids_r + eg_ids
                    segs_r = segs_r + eg_segs

                left_len = len(ids_l)
                right_len = len(ids_r)
                assert left_len == len(segs_l)
                assert right_len == len(segs_r)
                pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
                if left_len < self.max_left_len:
                    res = self.max_left_len - left_len  # residual
                    ids_l = ids_l + [pad_id] * res
                    segs_l = segs_l + [0] * res
                    att_mask_l = [1] * left_len + [0] * res
                else:
                    ids_l = ids_l[:self.max_left_len]
                    segs_l = segs_l[:self.max_left_len]
                    att_mask_l = [1] * self.max_left_len

                if right_len < self.max_right_len:
                    res = self.max_right_len - right_len
                    ids_r = ids_r + [pad_id] * res
                    segs_r = segs_r + [0] * res
                    att_mask_r = [1] * right_len + [0] * res
                else:
                    ids_r = ids_r[:self.max_right_len]
                    segs_r = segs_r[:self.max_right_len]
                    att_mask_r = [1] * self.max_right_len

                example = [ids_l, segs_l, att_mask_l, ids_r, att_mask_r, segs_r, label]

                examples.append(example)

                if (i + 1) % 10000 == 0:
                    print(f'{i + 1} sentences have been processed.')
            print(f'{file_dir} finished.')

        return examples


class VUA_All_Processor(Data_Processor):
    def __init__(self, args):
        super(VUA_All_Processor, self).__init__(args)

    def get_train_data(self):
        train_data_path = os.path.join(self.data_dir, 'VUA_All/train.csv')
        train_data = self._get_examples(train_data_path)
        return train_data

    def get_val_data(self):
        val_data_path = os.path.join(self.data_dir, 'VUA_All/val.csv')
        val_data = self._get_examples(val_data_path)
        return val_data

    def get_test_data(self):
        test_data_path = os.path.join(self.data_dir, 'VUA_All/test.csv')
        test_data = self._get_examples(test_data_path)
        return test_data

    def get_acad(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/acad.csv')
        data = self._get_examples(data_path)
        return data

    def get_conv(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/conv.csv')
        data = self._get_examples(data_path)
        return data

    def get_fict(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/fict.csv')
        data = self._get_examples(data_path)
        return data

    def get_news(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/genre/news.csv')
        data = self._get_examples(data_path)
        return data

    def get_adj(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/adj.csv')
        data = self._get_examples(data_path)
        return data

    def get_adv(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/adv.csv')
        data = self._get_examples(data_path)
        return data

    def get_noun(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/noun.csv')
        data = self._get_examples(data_path)
        return data

    def get_verb(self):
        data_path = os.path.join(self.data_dir, 'VUA_All/pos/verb.csv')
        data = self._get_examples(data_path)
        return data


class Verb_Processor(Data_Processor):
    def __init__(self, args):
        super(Verb_Processor, self).__init__(args)

    def get_train_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/train.csv')
        data = self._get_examples(data_path)
        return data

    def get_val_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/val.csv')
        data = self._get_examples(data_path)
        return data

    def get_test_data(self):
        data_path = os.path.join(self.data_dir, 'VUA_Verb/test.csv')
        data = self._get_examples(data_path)
        return data

    def get_trofi(self):
        data_path = os.path.join(self.data_dir, 'TroFi/TroFi.csv')
        data = self._get_examples(data_path)
        return data

    def get_mohx(self):
        data_path = os.path.join(self.data_dir, 'MOH-X/MOH-X.csv')
        data = self._get_examples(data_path)
        return data


def get_local(tokens, target_start, sep_puncs):
    """
    A local context is the clause that the target word occurs. Use sep_puncs to split different clauses.

    :param tokens: (list) a tokenized sentence
    :param target_start: (int) the start idx of the target_word
    :param sep_puncs: (list) all the punctuations that split a context
    :return: (tuple of int) the start idx and end idx of local context
    """
    local_start = 1
    local_end = local_start + len(tokens)
    for i, w in enumerate(tokens):
        if i < target_start and w in sep_puncs:
            local_start = i + 1
        if i > target_start and w in sep_puncs:
            local_end = i
            break
    return local_start, local_end


def target_align(target_position, sentence, tokenizer):
    """
    A target may be cut into word pieces by Tokenizer, this func tries to find the start and end idx of the target word
    after tokenization.
    NOTICE: we return a half-closed range. eg. [0, 6) for start_idx=0 and end_idx=6

    :param target_position: (int) the position of the target word in the original sentence
    :param sentence: (string) original sentence
    :param tokenizer: an instance of Transformers Tokenizer
    :return: (tuple of int) the start and end idx of the target word in the tokenized form
    """
    start_idx = 1  # take the [CLS] into consideration
    end_idx = 0
    for j, word in enumerate(sentence.split()):
        if not j == 0:
            word = ' ' + word
        word_tokens = tokenizer.tokenize(word)
        if not j == target_position:  # if current word is not target word, just count its length
            start_idx += len(word_tokens)
        else:  # else, calculate the end position
            end_idx = start_idx + len(word_tokens)
            break  # once find, stop looping.
    return start_idx, end_idx


if __name__ == '__main__':
    pass
