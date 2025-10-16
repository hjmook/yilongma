from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler
import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local imports (support running as script or module)
try:
    from .hybrid_model_client import HybridModelClient
    from .hybrid_config import MODE
except ImportError:
    # Running as a script: add current directory to sys.path and import absolutely
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hybrid_model_client import HybridModelClient
    from hybrid_config import MODE


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

CHOOSE_MODE = 1

client = HybridModelClient()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Welcome message
    welcome_msg = (
        "Hello! I'm Elon Musk (well, a chatbot version 😄)\n\n"
        "Ask me anything about:\n"
        "• Tesla, SpaceX, Neuralink, X (Twitter)\n"
        "• Technology, AI, space exploration\n"
        "• My thoughts and philosophies\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/stop - Stop our conversation\n"
        "/reset - Reset our conversation\n"
        "/health - Check model server status\n\n"
        "Let's chat!"
    )
    try:
        await update.message.reply_text(welcome_msg)
    except Exception:
        pass

    keyboard = [["elon-fast", "elon-detailed"]]
    await update.message.reply_text(
        "Choose a mode to begin:\n- ⚡ elon-fast: faster, may be less factually correct\n- 🧐 elon-detailed: slower, may be more factually correct\n\nYou can /stop when you're done.",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return CHOOSE_MODE


async def choose_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = (update.message.text or "").strip().lower()
    if choice not in ("elon-fast", "elon-detailed"):
        await update.message.reply_text("Please choose 'elon-fast' or 'elon-detailed'.")
        return CHOOSE_MODE

    # Start necessary local services based on choice
    os.environ["ELON_MODE"] = choice
    # Send waiting message first
    waiting_msg = await update.message.reply_text("⏳ Waiting for Elon to get ready...")
    if choice == "elon-fast":
        # Start local fast server
        import subprocess, time, requests
        server = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "fast_model_server.py")])
        context.chat_data["server_proc"] = server
        # Wait for health
        for _ in range(180):
            try:
                r = requests.get("http://localhost:5001/health", timeout=2)
                if r.status_code == 200 and r.json().get("ready"):
                    break
            except Exception:
                pass
            time.sleep(1)
        # Point client to fast server
        os.environ["HYBRID_MODEL_URL"] = "http://localhost:5001/predict"
        client.set_endpoint("http://localhost:5001/predict")
    else:
        # Start analyzer then detailed server
        import subprocess, time, requests
        analyzer = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "analyzer_service.py")])
        for _ in range(180):
            try:
                r = requests.get("http://localhost:6767/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        server = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "detailed_model_server.py")])
        context.chat_data["analyzer_proc"] = analyzer
        context.chat_data["server_proc"] = server
        for _ in range(240):
            try:
                r = requests.get("http://localhost:5055/health", timeout=2)
                if r.status_code == 200 and r.json().get("ready"):
                    break
            except Exception:
                pass
            time.sleep(1)
        os.environ["HYBRID_MODEL_URL"] = "http://localhost:5055/predict"
        client.set_endpoint("http://localhost:5055/predict")

    # Replace waiting message with ready confirmation
    try:
        await waiting_msg.edit_text(f"✅ {choice} is ready. Type your message.")
    except Exception:
        await update.message.reply_text(f"✅ {choice} is ready.")
    return ConversationHandler.END


async def reset_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    if client.reset_history(user_id):
        await update.message.reply_text("✅ Chat history reset! Let's start fresh.")
    else:
        await update.message.reply_text("⚠️ Could not reset chat history. The server might be down.")


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    h = client.health_check()
    if h.get("status") == "healthy":
        detail = []
        if "device" in h:
            detail.append(f"Device: {h['device']}")
        if "rag_available" in h:
            detail.append(f"RAG: {'Yes' if h['rag_available'] else 'No'}")
        if "analyzer" in h:
            detail.append(f"Analyzer: {h['analyzer']}")
        await update.message.reply_text("✅ Server healthy!\n" + "\n".join(detail))
    else:
        await update.message.reply_text(f"⚠️ Unhealthy: {h.get('error', 'unknown')}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    user_id = str(update.effective_user.id)
    # Log message to terminal
    logger.info(f"[User {user_id}] {user_message}")
    thinking_msg = await update.message.reply_text("Elon is thinking 🧠...")
    await update.message.chat.send_action(action="typing")
    resp = client.get_response(user_message, user_id=user_id, use_rag=True)
    try:
        await thinking_msg.edit_text(resp)
    except Exception:
        await update.message.reply_text(resp)
    logger.info(f"[Elon] {resp}")


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mode = os.environ.get("ELON_MODE", "unknown")
    waiting = await update.message.reply_text(
        f"⏳ Waiting for Elon to go home..."
    )
    # Terminate started processes
    for key in ["server_proc", "analyzer_proc"]:
        proc = context.chat_data.get(key)
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            context.chat_data[key] = None
    try:
        await waiting.edit_text("🏠 Elon has reached home")
    except Exception:
        await update.message.reply_text("🏠 Elon has reached home")




def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set. Please check your .env file.")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSE_MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_mode)],
        },
        fallbacks=[],
    )

    application.add_handler(conv)
    application.add_handler(CommandHandler("reset", reset_chat))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


