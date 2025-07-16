from flask import Flask, render_template, request
import joblib
from groq import Groq

import os
from dotenv import load_dotenv
import requests

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    q = request.form.get("q")
    # db
    return(render_template("main.html"))

@app.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@app.route("/llama_reply",methods=["GET","POST"])
def llama_reply():
    q = request.form.get("q")
    # load model
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content))

def llama_groq_response(q):
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": q}]
    )
    return completion.choices[0].message.content

@app.route("/deepseek",methods=["GET","POST"])
def deepseek():
    return(render_template("deepseek.html"))

@app.route("/deepseek_reply",methods=["GET","POST"])
def deepseek_reply():
    q = request.form.get("q")
    # load model
    client = Groq(api_key=groq_api_key)
    completion_ds = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("deepseek_reply.html",r=completion_ds.choices[0].message.content))

def deepseek_groq_response(q):
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": q}]
    )
    return completion.choices[0].message.content


@app.route("/dbs",methods=["GET","POST"])
def dbs():
    return(render_template("dbs.html"))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    q = float(request.form.get("q"))
    # load model
    model = joblib.load("dbs.jl")
    # make prediction
    pred = model.predict([[q]])
    return(render_template("prediction.html",r=pred))

def dbs_prediction(value):
    try:
        model = joblib.load("dbs.jl")
        pred = model.predict([[float(value)]])
        return f"Predicted DBS price: {pred[0]:.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

    
@app.route("/start_telegram", methods=["POST"])
def start_telegram():
    webhook_url = "https://dbss-telegram.onrender.com/telegram"  
    response = requests.get(f"{TELEGRAM_API}/setWebhook", params={"url": webhook_url})

    msg = " Telegram webhook set successfully." if response.ok else f" Failed: {response.text}"
    return render_template("main.html", message=msg)


@app.route("/stop_telegram", methods=["POST"])
def stop_telegram():
    response = requests.get(f"{TELEGRAM_API}/deleteWebhook")

    msg = " Telegram webhook deleted." if response.ok else f" Failed: {response.text}"
    return render_template("main.html", message=msg)
   
    
@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    data = request.get_json()

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"].strip()

        # Command routing
        if user_message.startswith("/llama"):
            query = user_message.replace("/llama", "", 1).strip()
            reply_text = llama_groq_response(query)

        elif user_message.startswith("/deepseek"):
            query = user_message.replace("/deepseek", "", 1).strip()
            reply_text = deepseek_groq_response(query)

        elif user_message.startswith("/dbs"):
            value = user_message.replace("/dbs", "", 1).strip()
            reply_text = dbs_prediction(value)

        else:
            reply_text = (
                "Please use one of the following commands:\n"
                "/llama <your question>\n"
                "/deepseek <your question>\n"
                "/dbs <numeric input>"
            )

        # Send reply back to Telegram chat
        requests.get(
            f"{TELEGRAM_API}/sendMessage",
            params={"chat_id": chat_id, "text": reply_text}
        )

    return "OK", 200

if __name__ == "__main__":
    app.run()
