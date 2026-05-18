import asyncio, os
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo
from dotenv import load_dotenv

load_dotenv()
TOKEN=os.getenv("BOT_TOKEN","PASTE_TOKEN")
WEBAPP_URL=os.getenv("WEBAPP_URL","https://YOUR_SITE.netlify.app")

bot=Bot(TOKEN)
dp=Dispatcher()

menu=ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="🔥 Открыть FaceOnMax", web_app=WebAppInfo(url=WEBAPP_URL))]],
    resize_keyboard=True
)

@dp.message(Command("start"))
async def start(m: Message):
    await m.answer("🔥 FaceOnMax V2\nТеперь анализ идёт по лицу через landmarks.", reply_markup=menu)

async def main():
    print("Bot started")
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
