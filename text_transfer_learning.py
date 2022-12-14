from fastai.text.all import *
import wandb
from fastai.callback.wandb import *

wandb.init(project='fastainlp')

path = untar_data(URLs.IMDB)

# get_text_files get all the text files in a path
files = get_text_files(path, folders=['train', 'test', 'unsup'])
# Logging
print("get text files \n")

# tokenize 1
spacy = WordTokenizer()
# Logging
print("load spacy \n")
# tokenize 2
tkn = Tokenizer(spacy)
# Logging
print("load tokenizer \n")

# first 2000 movie reviews
txts = L(o.open().read() for o in files[:2000])
toks200 = txts[:200].map(tkn)
# Logging
print("load first 2000 movie reviews \n")

# numericalize
num = Numericalize()
num.setup(toks200)
nums200 = toks200.map(num)
# Logging
print("done numericalize \n")

dl = LMDataLoader(nums200)
x, y = first(dl)
# logging
print("load LMDataLoader \n")

get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])
dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)

# Logging
print("load dls_lm \n")

# initial PLM
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    # perplexity() = torch.exp(cross_entropy), classification task, accuracy = the number of times the model is right at predicting the next word
    metrics=[accuracy, Perplexity()], cbs=WandbCallback()
).to_fp16()

# first phase fine-tuned LM
learn.fit_one_cycle(1, 2e-2)
learn.save('1epoch')
learn.load('1epoch')
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)
learn.save_encoder('finetuned')
print("save encoder \n")

dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab), CategoryBlock),
    get_y=parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)
print("load dls_clas \n")

dls_clas.show_batch(max_n=3)

learn = text_classifier_learner(
    dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy, cbs=WandbCallback()).to_fp16()

# load would raise an exception is an incomplete model is loaded, so we use load_encoder
learn = learn.load_encoder('finetuned')
# train with discriminative learning rate and gradual unfreezing, in NLP, unfreezing a few layers at a time has better perform
learn.fit_one_cycle(1, 2e-2)
learn.freeze_to(-2)  # free all except the last two parameter groups
# higher learning rate for final layers, lower learning rate for earlier
learn.fit_one_cycle(1, slice(1e-2/(2.6**-4), 1e-2))
learn.freeze_to(-3)  # unfree a bit more
# continue training at a lower learning rate
learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))
learn.freeze_to()  # unfree the whole model
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3))
