from model import TitleClassifier
from constants import MODEL_TITLES_SPACE_SIZE_CI

import gradio as gr


def get_title(path_to_data, save_path, num_of_books=None, num_of_batches=None):
    model = TitleClassifier(titles_list=MODEL_TITLES_SPACE_SIZE_CI)
    print(model)
    print(type(model))
    inference_df = model.get_titles(path_to_data, save_path, num_of_books, num_of_batches)
    print(inference_df)

    return inference_df


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
                                    '(Probably, you don`t have a clue so you can type "None")')
            path_to_data = gr.Textbox(label='Type the path to the books descriptions titles of which you want to get')
            save_path = gr.Textbox(label='Where the result should be saved?')
            num_of_books = gr.Textbox(label='How many books from the given set do you want to find? '
                                            '(The first N books will be found)')
        with gr.Row() as row:
            submit_button = gr.Button('Submit')
            submit_button.click(get_title, [path_to_data, save_path, num_of_books], answer)

    iface.launch(share=True)


if __name__ == '__main__':
    main()
