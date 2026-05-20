# 🎙️ Voice Assistant — Interview-Ready Documentation

> A modular AI Voice Assistant built with **Python**, **Flask**, **LangChain**, and **Groq (Llama 3)**. It handles Q&A, music streaming, real-time weather, and smart reminders — all driven by voice.

---

## 📌 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [How to Explain It in an Interview](#2-how-to-explain-it-in-an-interview)
3. [System Architecture](#3-system-architecture)
4. [Step-by-Step Working Flow](#4-step-by-step-working-flow)
5. [Module-by-Module Breakdown](#5-module-by-module-breakdown)
6. [AI Models & APIs Used](#6-ai-models--apis-used)
7. [Prompt Engineering Strategy](#7-prompt-engineering-strategy)
8. [Key Design Decisions (Interview Q&A)](#8-key-design-decisions-interview-qa)
9. [Technical Challenges & Solutions](#9-technical-challenges--solutions)
10. [Project File Structure](#10-project-file-structure)
11. [Installation & Setup](#11-installation--setup)
12. [API Endpoints](#12-api-endpoints)

---

## 1. What Is This Project?

This is a **full-stack, voice-driven AI assistant** that:

- Listens to the user's voice via the browser (Web Speech API)
- Converts speech to text and sends it to a Flask backend
- Uses **LLM-based intent detection** (Groq + Llama 3) to route the request
- Executes the correct module: Q&A, Music, Weather, or Reminder
- Speaks the response back using Text-to-Speech

**Four core capabilities:**

| Capability | What It Does |
|---|---|
| 💬 **Q&A** | General knowledge chatbot with conversation memory |
| 🎵 **Music** | Streams YouTube audio without downloading |
| 🌤️ **Weather** | Real-time weather using free Open-Meteo API |
| ⏰ **Reminder** | Sets background reminders using daemon threads |

---

## 2. How to Explain It in an Interview

### 🗣️ One-Line Pitch
> *"I built a modular AI voice assistant using Python and Flask where the user speaks into the browser, and the backend uses Meta's Llama 3 model via Groq's API to understand intent and route the query to the right module — for Q&A, music, weather, or reminders."*

### 🗣️ 30-Second Pitch
> *"The project has a clean, layered architecture. On the frontend, the browser's Web Speech API handles voice input and text-to-speech output. On the backend, Flask receives the transcribed text, and I use few-shot prompting with Llama 3 to detect user intent. Based on the intent — whether it's a question, music request, weather query, or reminder — it routes to a dedicated module. Each module is independent, making the system easy to extend."*

### 🗣️ Key Points to Highlight
- ✅ Used **Groq's inference API** for sub-second LLM responses (critical for voice UX)
- ✅ Used **few-shot prompting** instead of training a separate classifier
- ✅ **LangChain ConversationBufferMemory** for multi-turn Q&A
- ✅ **yt-dlp + VLC** for zero-download YouTube audio streaming
- ✅ **Daemon threads** for non-blocking background reminders
- ✅ Fully **modular architecture** — add new features by writing a new module

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────┐
│                 Browser (Frontend)               │
│   [Web Speech API → STT]  [SpeechSynthesis → TTS]│
└─────────────────────┬────────────────────────────┘
                      │  HTTP POST /process_input
                      ▼
┌──────────────────────────────────────────────────┐
│              Flask Backend (app.py)              │
│                                                  │
│   detect_intent() ◄─── Groq LLM (Llama 3)       │
│         │                                        │
│   ┌─────▼────────────────────────────────┐       │
│   │           Intent Router              │       │
│   └──┬──────┬──────────┬──────────┬──────┘       │
│      │      │          │          │               │
│   ┌──▼──┐ ┌─▼───┐ ┌───▼───┐ ┌───▼──────┐        │
│   │ QnA │ │Music│ │Weather│ │ Reminder │        │
│   └──┬──┘ └──┬──┘ └───┬───┘ └───┬──────┘        │
└──────┼────────┼─────────┼─────────┼──────────────┘
       │        │         │         │
    Groq     yt-dlp   Open-Meteo  Threading
    LLM      + VLC    REST APIs   + pyttsx3
```

---

## 4. Step-by-Step Working Flow

### Step 1 — User Opens the App
```
Browser → http://127.0.0.1:5006
→ Landing page asks for Groq API Key
→ User submits key via POST /
→ All modules initialized with this key
→ Redirected to /assistant chat UI
```

### Step 2 — User Speaks
```
User clicks "Start" mic button
→ Web Speech API (SpeechRecognition) activates
→ Voice converted to text in the browser
→ Text sent via: POST /process_input { "user_input": "play Shape of You" }
```

### Step 3 — Intent Detection
```python
# In app.py → detect_intent()
# Groq Llama 3 is called with 35 few-shot examples
# Returns one of: qna | music | weather | reminder | unknown

user_input = "play Shape of You"
intent = detect_intent(user_input)   # → "music"
current_functionality = "music"
```

### Step 4 — Module Routing
```python
if current_functionality == "qna":
    response = qna_module.answer(user_input)

elif current_functionality == "music":
    response = music_module.handle_play_music(user_input)

elif current_functionality == "weather":
    response = weather_module.get_weather(user_input)

elif current_functionality == "reminder":
    response = reminder_module.handle(user_input, conversation_context)
```

### Step 5 — Module Executes

**Music example:**
```
"play Shape of You"
  → Groq Llama 3 refines → "Shape of You Ed Sheeran"
  → yt-dlp searches YouTube → extracts direct audio stream URL
  → VLC MediaPlayer streams the audio
  → Returns: "Playing: Ed Sheeran - Shape Of You"
```

### Step 6 — Response Spoken Back
```
Flask returns JSON: { "response": "Playing: Ed Sheeran - Shape Of You" }
→ Browser receives it
→ SpeechSynthesis API speaks the response aloud
→ Response also displayed in chat UI
```

---

## 5. Module-by-Module Breakdown

### 📄 `app.py` — Flask Core

The central hub. Manages global state and routes all HTTP requests.

```python
# Global state
current_functionality = None    # "qna" | "music" | "weather" | "reminder"
conversation_context = {}       # Tracks multi-turn states (e.g., reminder flow)
groq_api_key = None             # Set at runtime, never saved to disk
```

---

### 💬 `modules/qna.py` — Q&A Module

Multi-turn conversational Q&A using LangChain.

```python
llm = ChatGroq(model="llama3-8b-8192", temperature=0.6)
memory = ConversationBufferMemory(memory_key="history", input_key="question")
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Every call automatically includes previous conversation history
response = chain.run(question=user_input)
```

**Interview point:** *"I used ConversationBufferMemory so the LLM remembers what was said earlier. For example, if the user asks 'Who is Einstein?' and then 'What did he discover?', the model understands 'he' refers to Einstein."*

---

### 🎵 `modules/music.py` — Music Module

Two-stage pipeline: LLM refinement → YouTube stream.

```
Stage 1: LLM Query Refinement
  "something chill from the 90s" → Groq (temp=0.4) → "90s chill mix"

Stage 2: Fetch & Stream
  "90s chill mix"
  → yt_dlp.extract_info("ytsearch:90s chill mix", download=False)
  → Gets direct audio stream URL
  → vlc.MediaPlayer(url).play()
```

**Interview point:** *"Files are never downloaded. yt-dlp extracts the stream URL, and VLC plays it directly — zero disk usage."*

---

### 🌤️ `modules/weather.py` — Weather Module

Three-stage pipeline: LLM extracts city → Geocoding → Weather.

```
"Is it raining in Tokyo?"
  → Groq Llama 3 → "Tokyo"
  → Open-Meteo Geocoding API → lat: 35.68, lon: 139.69
  → Open-Meteo Forecast API → temp: 22°C, code: 61
  → weather_description_fetcher(61) → "Slight rain"
  → "Tokyo: Slight rain, 22°C, windspeed 12 km/h"
```

---

### ⏰ `modules/reminder.py` — Reminder Module

A mini state machine with 3 conversation turns, backed by daemon threads.

```
Turn 1: "Remind me to take medicine"
  → pending_action = "set_reminder_time"
  → Reply: "What time should I set the reminder for?"

Turn 2: "8 PM" → normalize_time() → "08:00 PM"
  → pending_action = "set_reminder_message"
  → Reply: "What should I remind you about?"

Turn 3: "Take blood pressure tablet"
  → threading.Thread(target=reminder_thread, daemon=True).start()
  → Reply: "Reminder set for 08:00 PM: 'Take blood pressure tablet'"
```

```python
# Background thread checks time every second
def reminder_thread():
    while True:
        if datetime.now().strftime("%I:%M %p") == reminder_time:
            speech_module.speak(f"Reminder: {message}")
            break
        time.sleep(1)
```

---

### 🔧 `modules/utils.py` — Shared Utilities

| Function | Purpose |
|---|---|
| `normalize_time(input)` | Regex-based: `"854pm"` → `"08:54 PM"` |
| `extract_location(input, llm)` | Few-shot LLM: extracts city from weather query |
| `weather_description_fetcher(code)` | Maps WMO weather codes to readable strings |

---

### 🗣️ `modules/speech.py` — Speech Module

- **TTS**: `pyttsx3` runs in a separate thread (200 wpm rate)
- **Interrupt**: Calls `engine.stop()` + `thread.join()` before new speech
- **STT (backend fallback)**: `SpeechRecognition` + `PyAudio`

---

## 6. AI Models & APIs Used

### 🤖 LLM: Meta Llama 3 (via Groq)

| Attribute | Value |
|---|---|
| **Model** | `llama3-8b-8192` |
| **Provider** | Groq Cloud |
| **Context Window** | 8,192 tokens |
| **Library** | `langchain_groq.ChatGroq` |

Used in 4 ways with different temperatures:

| Usage | Temperature | Task |
|---|---|---|
| Intent Detection | 0.6 | Classify: qna / music / weather / reminder |
| Q&A Chain | 0.6 | Answer questions with memory |
| Music Refinement | 0.4 | Extract clean song search query |
| Location Extraction | 0.7 | Extract city name from weather query |

---

### 🌍 Open-Meteo APIs (Free, No Key Required)

**Geocoding API:**
```
GET https://geocoding-api.open-meteo.com/v1/search?name=Tokyo
→ Returns: latitude, longitude, city name
```

**Weather Forecast API:**
```
GET https://api.open-meteo.com/v1/forecast?latitude=35.68&longitude=139.69&current_weather=true
→ Returns: temperature (°C), windspeed (km/h), weathercode (WMO)
```

---

### 🎬 YouTube (via yt-dlp, No Key Required)

```python
ydl_opts = { 'format': 'bestaudio/best', 'noplaylist': True, 'quiet': True }
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    results = ydl.extract_info(f"ytsearch:{query}", download=False)
    stream_url = results['entries'][0]['url']   # Direct audio — no download
```

---

### 🌐 Browser APIs (No Key, No Server)

| API | Used For |
|---|---|
| `SpeechRecognition` | Voice → Text (STT) |
| `SpeechSynthesis` | Text → Voice (TTS) |

---

## 7. Prompt Engineering Strategy

### Intent Detection (`prompts/main_prompt.py`)

- **Type**: Few-shot classification
- **Examples**: 35 labeled pairs (input → intent)
- **Output**: Single word — `qna`, `music`, `weather`, `reminder`, `unknown`

```python
# Sample examples fed to Llama 3
{"input": "What is machine learning?",       "intent": "qna"},
{"input": "Play some relaxing piano music.", "intent": "music"},
{"input": "What's the weather in NYC?",      "intent": "weather"},
{"input": "Remind me to call Sarah at 4 PM","intent": "reminder"},
```

### Q&A Prompt (`prompts/qna_prompts.py`)
- LangChain `PromptTemplate` with `{history}` + `{question}` variables
- Memory injected automatically on every call

### Music Prompt (`prompts/music_prompt.py`)
- 30 examples covering English, Hindi, Telugu, Tamil
- Extracts clean search string, no extra text

### Location Prompt (`prompts/location_prompt.py`)
- 9 examples for city extraction from natural language

---

## 8. Key Design Decisions (Interview Q&A)

**Q: Why Groq instead of OpenAI?**
> Groq provides near-instant inference (sub-second latency) on Llama 3, which is critical for voice UX — any delay feels unnatural to the user.

**Q: Why yt-dlp instead of the YouTube Data API?**
> The official API has strict quotas and doesn't give stream URLs. `yt-dlp` extracts direct audio streams without downloading files — enabling true zero-latency playback via VLC.

**Q: Why Open-Meteo instead of OpenWeatherMap?**
> Open-Meteo is completely free with no API key, making the app accessible to everyone. It uses ECMWF model data and is highly accurate.

**Q: Why few-shot prompting for intent detection instead of a classifier?**
> Training a classifier needs labeled data and retraining. Few-shot prompting lets Llama 3 do in-context learning — adding a new intent just means adding a few examples to the prompt file. Zero retraining.

**Q: Why daemon threads for reminders?**
> Daemon threads automatically die when the main process exits, preventing zombie threads from running after the app stops.

**Q: Why not store the API key in a `.env` file?**
> The key is entered at runtime via the UI and stored only in server memory. It's cleared when the user clicks "Stop". No risk of accidentally committing secrets to version control.

---

## 9. Technical Challenges & Solutions

| Challenge | Solution |
|---|---|
| LLM latency in real-time voice | Used Groq (fastest available inference, < 1s) |
| Natural language time parsing | Custom regex normalizer: `"854pm"` → `"08:54 PM"` |
| Multi-turn reminder conversation | Mini state machine with `conversation_context` dict |
| Music interruption mid-play | `vlc.MediaPlayer.stop()` called from `/interrupt` endpoint |
| TTS interruption | `pyttsx3.stop()` + `thread.join()` before new speech starts |
| YouTube stream without download | `yt-dlp` with `download=False` + direct URL to VLC |
| Fuzzy music queries | LLM (temp=0.4) normalizes vague input to clean search terms |

---

## 10. Project File Structure

```
Voice_Assistant-main/
│
├── app.py                   # Flask routes, intent detection, global state
├── main.py                  # Entry point — runs Flask on port 5006
├── requirements.txt         # All Python dependencies
├── Dockerfile               # Docker container configuration
│
├── modules/                 # One file per feature
│   ├── qna.py               # LLMChain + ConversationBufferMemory
│   ├── music.py             # yt-dlp + VLC + Groq query refinement
│   ├── weather.py           # Open-Meteo API + location extraction
│   ├── reminder.py          # Threading + state machine
│   ├── speech.py            # pyttsx3 TTS + SpeechRecognition STT
│   └── utils.py             # normalize_time, extract_location, weather_description_fetcher
│
├── prompts/                 # All LLM prompt templates
│   ├── main_prompt.py       # Intent detection (35 few-shot examples)
│   ├── qna_prompts.py       # Q&A PromptTemplate with history
│   ├── music_prompt.py      # Music query extraction (30 examples)
│   └── location_prompt.py   # Location extraction (9 examples)
│
├── templates/               # HTML pages
│   ├── api_key.html         # Landing page — API key form
│   └── assistant.html       # Main voice assistant UI
│
├── static/                  # CSS, JS, images
└── logger/                  # Custom logging module
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.11+
- VLC Media Player installed
- Free Groq API key → [console.groq.com](https://console.groq.com)

### Step 1 — Clone the Repo
```bash
git clone <repository-url>
cd Voice_Assistant-main
```

### Step 2 — Create Virtual Environment
```bash
# Conda (recommended)
conda create -p ./env python=3.11 -y
conda activate ./env

# OR venv
python3.11 -m venv env
source env/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install VLC (Linux only)
```bash
sudo apt-get install vlc libvlc-dev
```

### Step 5 — Run the App
```bash
python main.py
```

### Step 6 — Open in Browser
```
http://127.0.0.1:5006
```

Enter your **Groq API Key** on the landing page → click **Submit** → start talking!

---

### 🐳 Docker (Optional)
```bash
docker build -t voice-assistant .
docker run -p 5006:5006 voice-assistant
# Open: http://localhost:5006
```

---

## 12. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET/POST` | `/` | Landing page — collect Groq API key |
| `GET` | `/assistant` | Main voice chat UI |
| `POST` | `/process_input` | Route voice input to correct module |
| `POST` | `/start` | Initialize session, speak greeting |
| `POST` | `/interrupt` | Stop music or TTS immediately |
| `POST` | `/exit` | Reset module state, stop music |
| `POST` | `/stop` | Full reset — clear API key, go to landing |

**Sample `/process_input`:**
```json
// Request
{ "user_input": "What is the weather in Mumbai?" }

// Response
{ "response": "Mumbai: Partly cloudy, 31°C, windspeed 18 km/h." }
```

---

## 🔐 Security Notes

- API key lives **only in server memory** — never saved to disk or `.env`
- Clicking **Stop** clears the key from memory (`groq_api_key = None`)
- No conversation history is persisted after the session ends

---

*Built with ❤️ using Python, Flask, LangChain, Groq Llama 3, yt-dlp, and Web Speech API.*