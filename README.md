# AI-Powered Real-Time Voice Assistant

**Status: 🚧 Under Maintenance 🚧**  
This project is currently under development. We are actively working to enhance its features and functionality.

---

## 🎯 Project Overview

This project leverages advanced AI models to create an interactive real-time voice assistant. It integrates cutting-edge technologies to transcribe speech, generate intelligent responses, and synthesize natural-sounding speech.

### Key Features:
- **Real-Time Speech-to-Text**: Converts user speech into text using Whisper STT.
- **AI-Generated Responses**: Generates intelligent, contextual responses with advanced conversational AI.
- **Text-to-Speech**: Uses Coqui TTS to convert text responses back into speech for real-time interaction.
- **Voice Activity Detection**: Efficiently detects when the user is speaking using WebRTC VAD.

---

## 🚀 Future Goals and Improvements

1. **Mock Interview Mode**:
   - Develop an interactive mode where the AI acts as an interviewer.
   - Tailored questions based on the user's selected role or domain.
   - Provide personalized feedback on responses.

2. **Real-Time Feedback**:
   - Implement NLP-based analysis to evaluate the user's speech clarity, relevance, and tone.

3. **Domain-Specific Interviews**:
   - Add role-specific question sets for industries like technology, finance, and healthcare.

4. **Gamification and Progress Tracking**:
   - Include scoring mechanisms and badges to motivate users.
   - Track user performance over time.

5. **Deployment**:
   - Build a user-friendly web interface for global access.
   - Deploy a live version using Flask or FastAPI as the backend.

6. **Enhanced TTS Voices**:
   - Integrate multiple voices for better personalization and a more engaging experience.

---

## 🛠️ Tech Stack

- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Text-to-Speech**: [Coqui TTS](https://github.com/coqui-ai/TTS)
- **Voice Activity Detection**: [WebRTC VAD](https://webrtc.org/)
- **Conversational AI**: Ollama GPT Models
- **Programming Language**: Python
- **Libraries**: PyAudio, NumPy, SimpleAudio

---

## 📝 How to Run Locally

1. Clone the repository:
   ```bash
   git clone <repository-url>
Navigate to the project directory:
bash
Copy code
cd <project-directory>
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the application:
bash
Copy code
python main.py
📜 License
This project is licensed under the MIT License.

💬 Feedback and Suggestions
We welcome all feedback and suggestions! Feel free to open an issue or create a pull request.
