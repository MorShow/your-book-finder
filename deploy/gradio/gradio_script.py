from model import BookFinder

from functools import partial
from typing import Optional

import gradio as gr


def get_title(description: str, save_path: Optional[str] = None, model: Optional[BookFinder] = None):
    inference_df = model.predict(description, save_path)
    return_string = ''
    counter = 1

    for _, item in inference_df.iterrows():
        return_string += f'{counter}. book: {item.get("title")},\nprobability: {item.get("score") * 100:.2f} %\n'
        counter += 1

    return return_string


def main(model: BookFinder):
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
            save_path = gr.Textbox(label='Where the result should be saved? ( (/.../)<filename>.csv )')
            # TODO: (maybe) make this option available, but I`m not sure, it will bear optimization problems
            # num_of_books = gr.Textbox(label='How many books from the given set do you want to find? '
            #                                 '(The first N books will be found)')
        with gr.Row() as row:
            submit_button = gr.Button('Submit')
            submit_button.click(partial(get_title, model=model), [description, save_path], answer)

    iface.launch(share=True)


if __name__ == '__main__':
    book_finder = BookFinder(
        cluster_selection_epsilon=0.2546331876872773,
        k_neighbours_inference=29,
        min_cluster_size=17,
        umap_metric='euclidean',
        umap_min_dist=0.21872587813809782,
        umap_neighbors=8
    )
    book_finder.fit(r'C:\Users\aleks\OneDrive\Desktop\Studying\your-book-finder\data\raw\gutenberq_books_tiny.csv')

    main(book_finder)
