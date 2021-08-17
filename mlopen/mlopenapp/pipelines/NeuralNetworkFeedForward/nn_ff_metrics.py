from sklearn.metrics import classification_report
import torch
import numpy


def test_model(X_test, y_test, review_dict):
    bow_ff_nn_predictions = []
    original_lables_ff_bow = []
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec = make_bow_vector(review_dict, row['stemmed_tokens'])
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(make_target(Y_test['sentiment'][index]).cpu().numpy()[0])
    print(classification_report(original_lables_ff_bow,bow_ff_nn_predictions))
    ffnn_loss_df = pd.read_csv(ffnn_loss_file_name)
    print(len(ffnn_loss_df))
    print(ffnn_loss_df.columns)
    ffnn_plt_500_padding_100_epochs = ffnn_loss_df[' loss'].plot()
    fig = ffnn_plt_500_padding_100_epochs.get_figure()
    fig.savefig(OUTPUT_FOLDER + 'plots/' + "ffnn_bow_loss_500_padding_100_epochs_less_lr.pdf")
