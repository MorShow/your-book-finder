from model import BookFinder

import asyncio
from collections import defaultdict
from functools import partial

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters, ConversationHandler)

BOT_TOKEN = '...'
states_dict = defaultdict(asyncio.Lock)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm your bot 🤖 You can type the description of some book"
                                    "and get a recommendation from me. I will try to match your interests"
                                    "as close as possible")


async def give_recommendation(update: Update, context: ContextTypes.DEFAULT_TYPE, model: BookFinder):
    user_id = update.effective_user.id
    lock = states_dict[user_id]

    locked = await lock.acquire()
    if not locked:
        await update.message.reply_text("I`m still processing your request")
        return

    try:
        await update.message.reply_text("Please, wait for a response a little..")
        await update.message.reply_text(f"Hmmm... I think you might looking for {model.predict(update.message.text)}")
    except Exception as e:
        print(e)
        await update.message.reply_text("Sorry, something went wrong. Try again later")
    finally:
        lock.release()


if __name__ == '__main__':
    book_finder = BookFinder(cluster_selection_epsilon=0.2546331876872773,
                             k_neighbours_inference=29,
                             min_cluster_size=17,
                             umap_metric='euclidean',
                             umap_min_dist=0.21872587813809782,
                             umap_neighbors=8
                             )
    book_finder.fit(r"...")
    give_recommendation_wrapper = partial(give_recommendation, model=book_finder)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, give_recommendation_wrapper))
    app.run_polling()
