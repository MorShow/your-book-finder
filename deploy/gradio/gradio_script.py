from model import TitleClassifier
from constants import MODEL_TITLES_SIZE_FAST_TEST, MODEL_TITLES_SIZE_CI, NUM_OF_BATCHES_CI

from functools import partial

import gradio as gr


def get_title(path_to_data, save_path, num_of_books=None, num_of_batches=None):
    model = TitleClassifier(titles_list_arg=MODEL_TITLES_SIZE_FAST_TEST)

    inference_df = model.get_titles(path_to_data, save_path, int(num_of_books), num_of_batches)
    return_string = ''
    counter = 1

    for item in inference_df.iterrows():
        print(item[1]['scores'])
        scores_dict = item[1]['scores']
        if isinstance(item[1]['scores'], str):
            scores_dict = eval(item[1]['scores'])
        labels = list(scores_dict.keys())
        values = list(scores_dict.values())
        max_likelihood = max(values)
        max_likelihood_label = labels[values.index(max_likelihood)]

        return_string += f'{counter}. book: {max_likelihood_label}, {round(max_likelihood * 100, 2)}%\n'
        counter += 1

    return return_string


wrapper = partial(get_title, num_of_batches=NUM_OF_BATCHES_CI)


def main():
    with gr.Blocks() as iface:
        with gr.Row() as row:
            gr.HTML('<h1>Your book finder</h1>')
        with gr.Row() as row:
            gr.HTML('<p>Type the description of the book you want to find.</p>')
        with gr.Row() as row:
            answer = gr.Text()
        with gr.Row() as row:
            description = gr.Textbox(label='Description')
            language = gr.Textbox(label='The language of the book in which it was written')
            year = gr.Textbox(label='The year the book was written '
                                    '(Probably, you do not have even a clue so you can just type "None")')
            path_to_data = gr.Textbox(label='Type the path to the books descriptions titles of which you want to get')
            save_path = gr.Textbox(label='Where the result should be saved?')
            num_of_books = gr.Textbox(label='How many books from the given set do you want to find? '
                                            '(The first N books will be found)')
        with gr.Row() as row:
            submit_button = gr.Button('Submit')
            submit_button.click(wrapper, [path_to_data, save_path, num_of_books], answer)

    iface.launch(share=True)


if __name__ == '__main__':
    main()
