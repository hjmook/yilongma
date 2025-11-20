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

# Ping command to check if the current model server is online
import requests as _requests
async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    endpoint = getattr(client, 'endpoint', None) or os.environ.get("HYBRID_MODEL_URL")
    if not endpoint:
        await update.message.reply_text("❌ No model server endpoint configured.")
        return
    # Try to get the /health endpoint
    health_url = endpoint.replace("/predict", "/health")
    try:
        r = _requests.get(health_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", "unknown")
            await update.message.reply_text(f"✅ Server is online! Status: {status}")
        else:
            await update.message.reply_text(f"⚠️ Server responded with status code {r.status_code}")
    except Exception as e:
        await update.message.reply_text(f"❌ Server is offline or unreachable.\nError: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If already in an active session (mode selected and ready), ignore /start
    if context.user_data.get("ready_since") is not None:
        await update.message.reply_text("Already chatting! Use /stop to end the conversation first.")
        return ConversationHandler.END
    
    # Clear any existing state to ensure clean start
    context.user_data.clear()
    
    # Welcome message
    welcome_msg = (
        "Hello! I'm Elon Musk (well, a chatbot version)\n\n"
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

    keyboard = [["elon-fast", "elon-thinking"]]
    await update.message.reply_text(
        "Choose a mode to begin:\n- ⚡ elon-fast: faster, may be less factually correct\n- 🧐 elon-thinking: slower, may be more factually correct\n\nYou can /stop when you're done.",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return CHOOSE_MODE


async def choose_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = (update.message.text or "").strip().lower()
    if choice not in ("elon-fast", "elon-thinking"):
        await update.message.reply_text("Please choose 'elon-fast' or 'elon-thinking'.")
        return CHOOSE_MODE

    # Set flag to ignore messages during setup and record when setup started
    context.user_data["is_setting_up"] = True
    context.user_data["setup_start_time"] = update.message.date
    
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
        # Start analyzer then thinking server
        import subprocess, time, requests
        logger.info("Starting analyzer service...")
        analyzer = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "analyzer_service.py")])
        analyzer_ready = False
        for i in range(180):
            try:
                r = requests.get("http://localhost:6767/health", timeout=2)
                if r.status_code == 200:
                    logger.info(f"Analyzer ready after {i+1} seconds")
                    analyzer_ready = True
                    break
            except Exception as e:
                if i % 10 == 0:  # Log every 10 seconds
                    logger.info(f"Waiting for analyzer... ({i}s)")
            time.sleep(1)
        
        if not analyzer_ready:
            logger.error("Analyzer failed to start!")
            await waiting_msg.edit_text("❌ Analyzer failed to start. Please try again.")
            analyzer.terminate()
            context.user_data["is_setting_up"] = False
            return ConversationHandler.END
        
        logger.info("Starting thinking model server...")
        await waiting_msg.edit_text("⏳ Analyzer ready! Loading thinking model (this takes 30-60 seconds)...")
        server = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "thinking_model_server.py")])
        context.chat_data["analyzer_proc"] = analyzer
        context.chat_data["server_proc"] = server
        
        server_ready = False
        for i in range(240):
            logger.info(f"[Loop {i}] Checking health...")
            try:
                r = requests.get("http://localhost:5055/health", timeout=10)  # Increased from 2 to 10 seconds
                logger.info(f"[Loop {i}] Got response: {r.status_code}")
                if r.status_code == 200:
                    health_data = r.json()
                    logger.info(f"[Loop {i}] Health: ready={health_data.get('ready')}, status={health_data.get('status')}")
                    if health_data.get("ready"):
                        logger.info(f"✅ Thinking server ready after {i+1} checks!")
                        server_ready = True
                        break
            except requests.exceptions.ConnectionError as e:
                logger.info(f"[Loop {i}] ConnectionError - server not started yet")
            except Exception as e:
                logger.warning(f"[Loop {i}] Unexpected error: {type(e).__name__}: {e}")
            time.sleep(1)
        
        if not server_ready:
            logger.error("Thinking server failed to become ready!")
            await waiting_msg.edit_text("❌ Thinking server failed. Try 'elon-fast' instead or restart.")
            analyzer.terminate()
            server.terminate()
            context.user_data["is_setting_up"] = False
            return ConversationHandler.END
        
        os.environ["HYBRID_MODEL_URL"] = "http://localhost:5055/predict"
        client.set_endpoint("http://localhost:5055/predict")

    # Clear setup flag and record when bot became ready
    context.user_data["is_setting_up"] = False
    import datetime
    context.user_data["ready_since"] = datetime.datetime.now(datetime.timezone.utc)
    
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
    message_time = update.message.date
    
    # Ignore messages if bot is not ready yet
    if "ready_since" not in context.user_data:
        return
    
    # Ignore messages that were sent before bot became ready
    if message_time < context.user_data["ready_since"]:
        logger.info(f"[Ignored] Message sent before bot was ready: {update.message.text[:50]}")
        return
    
    # Ignore messages sent during ANY previous thinking period
    if "last_thinking_end_time" in context.user_data:
        # Check if message was sent during the last thinking period
        if "thinking_start_time" in context.user_data:
            thinking_start = context.user_data["thinking_start_time"]
            thinking_end = context.user_data["last_thinking_end_time"]
            if thinking_start <= message_time <= thinking_end:
                logger.info(f"[Ignored] Message sent during previous thinking period: {update.message.text[:50]}")
                return
    
    # If bot is currently thinking, ignore this message
    if context.user_data.get("is_thinking", False):
        logger.info(f"[Ignored] Message sent while bot is currently thinking: {update.message.text[:50]}")
        return
    
    user_message = update.message.text
    user_id = str(update.effective_user.id)
    
    # Set thinking flag and record when thinking started
    context.user_data["is_thinking"] = True
    import datetime
    context.user_data["thinking_start_time"] = datetime.datetime.now(datetime.timezone.utc)
    
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
    
    # Record when thinking ended and clear thinking flag
    context.user_data["last_thinking_end_time"] = datetime.datetime.now(datetime.timezone.utc)
    context.user_data["is_thinking"] = False


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    mode = os.environ.get("ELON_MODE", "unknown")
    user_id = str(update.effective_user.id)
    waiting = await update.message.reply_text(
        f"⏳ Waiting for Elon to go home..."
    )
    
    # End chat session for comprehensive logging (both modes now support this)
    try:
        import requests
        if mode == "elon-thinking":
            requests.post("http://localhost:5055/end_session", json={"user_id": user_id}, timeout=5)
        elif mode == "elon-fast":
            requests.post("http://localhost:5001/end_session", json={"user_id": user_id}, timeout=5)
    except Exception as e:
        logger.warning(f"Failed to end chat session: {e}")
    
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
    
    # Clear user state
    context.user_data.clear()
    
    try:
        await waiting.edit_text("🏠 Elon has reached home", reply_markup=ReplyKeyboardRemove())
    except Exception:
        await update.message.reply_text("🏠 Elon has reached home", reply_markup=ReplyKeyboardRemove())
    
    # End conversation handler
    return ConversationHandler.END




def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set. Please check your .env file.")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSE_MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_mode)],
        },
        fallbacks=[CommandHandler("stop", stop), CommandHandler("start", start)],
        allow_reentry=True,
    )

    application.add_handler(conv)
    application.add_handler(CommandHandler("reset", reset_chat))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(CommandHandler("ping", ping))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


