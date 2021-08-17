from input import text_files_input as tfi
import vectorization as vct,\
    text_preprocessing as tpp, \
    logistic_regression as lr, \
    frequencies as frq

"""
Function to calculate pos-neg frequencies
"""


def get_model(*args, verbose=False):
    print("Preparing Data. . .")
    df = tfi.prepare_data()
    tfidf = {}
    vector = {}
    for arg in args:
        print("For column " + str(arg) + ":")
        print("Processing Text. . .")
        df = tpp.process_text_df(df, arg)
        print("Create Corpus. . .")
        corpus = []
        for i, corp in df[arg].items():
            corpus.append(corp)
        print("Create Vector. . .")
        tfidf[arg], vector[arg] = vct.tf_idf(corpus)
        #freqs = frq.build_freqs(corpus, df['sentiment'].tolist())
        #x_pn = [frq.statement_to_freq(txt, freqs) for txt in corpus]
        #x_posneg = frq.get_posneg(corpus, freqs)
        s_a_model = lr.log_reg(vector[arg], df['sentiment'].tolist())
        return tfidf[arg], vector[arg], s_a_model
