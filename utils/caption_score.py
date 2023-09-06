import pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

if __name__ == '__main__':
    ref = "This is my house"
    ref_toekn = PTBTokenizer.tokenize(ref)
    pred = "This is your house"
    pred_token = PTBTokenizer.tokenize(pred)
    print(pred_token)
    print(dir(pycocoevalcap))
    scorer = Bleu(4)
    scorer.calc_score(ref_toekn, pred_token)