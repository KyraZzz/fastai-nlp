from fastai.text.all import *

path = untar_data(URLs.IMDB)

# get_text_files get all the text files in a path
files = get_text_files(path, folders=['train', 'test', 'unsup'])
# Logging
txt = files[0].open().read()
print(txt[:75])

# tokenize 1
spacy = WordTokenizer()
# Logging
toks = first(spacy([txt]))
print(coll_repr(toks, 30))
# tokenize 2
tkn = Tokenizer(spacy)
# Logging
print(coll_repr(tkn(txt), 31))

# first 2000 movie reviews
txts = L(o.open().read() for o in files[:2000])
toks200 = txts[:200].map(tkn)
# Logging
print(toks200[0])

# numericalize
num = Numericalize()
num.setup(toks200)
nums200 = toks200.map(num)
# Logging
nums = num(toks)[:50]
print(nums)
print(' '.join(num.vocab[o] for o in nums))

dl = LMDataLoader(nums200)
x, y = first(dl)
# logging
print(x.shape, y.shape)
print(' '.join(num.vocab[o] for o in x[0][:20]))
print(' '.join(num.vocab[o] for o in y[0][:20]))

get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])
dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)

# Logging
dls_lm.show_batch(max_n=2)

# initial PLM
learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    # perplexity() = torch.exp(cross_entropy), classification task, accuracy = the number of times the model is right at predicting the next word
    metrics=[accuracy, Perplexity()]
).to_fp16()

# first phase fine-tuned LM
learn.fit_one_cycle(10, 2e-2)
learn.save_encoder('finetuned')

dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab), CategoryBlock),
    get_y=parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)

dls_clas.show_batch(max_n=3)

learn = text_classifier_learner(
    dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()

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
