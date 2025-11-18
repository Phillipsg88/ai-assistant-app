📌 Phillips' Personalized AI Assistant

An open-source, multimodal AI assistant that writes high-end code, solves ICT problems, supports voice and image input, remembers user preferences, and is trainable on custom data.

---

🚀 Features

- 💻 Code Generation: Writes and explains complex code using DeepSeek Coder or Mixtral
- 🛠️ ICT Problem Solving: Diagnoses and suggests solutions using Qwen or Solar
- 🗣️ Voice Input: Transcribes speech using Whisper
- 🖼️ Image Input: Accepts screenshots or diagrams (LLaVA-ready)
- 🧠 Memory: Stores and retrieves past interactions using FAISS
- 🎯 Personalization: Adapts responses based on user profile
- 📚 Trainable: Fine-tune with your own data using QLoRA or Axolotl
- 🌐 Web App: Built with Streamlit for easy deployment

---

🧪 Tech Stack

| Layer              | Tools Used                                  |
|--------------------|---------------------------------------------|
| Language Models    | DeepSeek Coder, Qwen 2.5, Mixtral, Solar     |
| Voice Processing   | Whisper                                     |
| Image Understanding| LLaVA (optional)                            |
| Memory             | FAISS + LangChain                           |
| UI                 | Streamlit                                   |
| Deployment         | Streamlit Cloud, Render, Railway            |

---

📦 Installation

`bash
git clone https://github.com/your-username/ai-assistant-app.git
cd ai-assistant-app
pip install -r requirements.txt
streamlit run app.py
`

---

🌐 Deploy to Streamlit Cloud

1. Fork or clone this repo
2. Go to Streamlit Cloud
3. Connect your GitHub account
4. Select this repo and set app.py as the main file
5. Click Deploy and get your public link

---

🧠 Customization

- Modify user_profile in app.py to personalize tone, language, and interests
- Add new models or swap existing ones using Hugging Face
- Extend memory with long-term storage or user-specific profiles

---

📚 Training

To fine-tune your assistant:
- Format your data in Alpaca-style instruction-response pairs
- Use QLoRA or Axolotl for efficient training
- Upload new weights to Hugging Face or host locally

---

🤝 Contributing

Pull requests are welcome! Feel free to:
- Add new features
- Improve model routing
- Enhance UI/UX
- Expand training tools

---

📄 License

This project is open-source under the MIT License.