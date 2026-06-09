import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceLang, setVoiceLang] = useState("en-IN");
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input) return;

    const userText = input;

    setMessages((prev) => [...prev, { sender: "You", text: userText }]);
    setInput("");

    // typing indicator
    setMessages((prev) => [...prev, { sender: "Bot", text: "Typing..." }]);

    try {
      const res = await fetch("https://nonlimitative-nancie-credulously.ngrok-free.dev", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ message: userText }),
      });

      const data = await res.json();

      setMessages((prev) => {
        const updated = [...prev];
        updated.pop();
        return [...updated, { sender: "Bot", text: data.reply }];
      });
    } catch (error) {
      console.error("Error connecting to backend:", error);
      setMessages((prev) => {
        const updated = [...prev];
        updated.pop();
        return [...updated, { sender: "Bot", text: "Sorry, I couldn't connect to the server." }];
      });
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  const handleVoiceSearch = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech Recognition is not supported in this browser. Please try Google Chrome.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = voiceLang;
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => setIsRecording(true);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
    };

    recognition.onerror = (event) => {
      console.error("Speech Recognition Error:", event.error);
      setIsRecording(false);
    };

    recognition.onend = () => setIsRecording(false);

    recognition.start();
  };

  return (
    <>
      <div className="app-background">

        {/* Floating Chat Button */}
        {!isOpen && (
          <div className="chat-fab" onClick={() => setIsOpen(true)}>
            <span className="fab-icon">🌾</span>
          </div>
        )}

        {isOpen && (
          <div className="chat-overlay" onClick={() => setIsOpen(false)}>

            <div className="chat-window" onClick={(e) => e.stopPropagation()}>

              {/* Header */}
              <div className="chat-header">
                <div className="header-left">
                  <span className="bot-icon">🚜</span>
                  <div>
                    <h3>Kisan Sahayak</h3>
                    <p>AI Farming Assistant</p>
                  </div>
                </div>
                <button className="close-btn" onClick={() => setIsOpen(false)}>
                  ✖
                </button>
              </div>

              {/* Voice Language Selection */}
              <div className="voice-lang-bar">
                <span>🗣️ Voice Language:</span>
                <select value={voiceLang} onChange={(e) => setVoiceLang(e.target.value)}>
                  <option value="en-IN">English (India)</option>
                  <option value="hi-IN">Hindi (हिंदी)</option>
                  <option value="mr-IN">Marathi (मराठी)</option>
                  <option value="ta-IN">Tamil (தமிழ்)</option>
                  <option value="te-IN">Telugu (తెలుగు)</option>
                  <option value="pa-IN">Punjabi (ਪੰਜਾਬੀ)</option>
                </select>
              </div>

              {/* Chat Body */}
              <div className="chat-body">
                {messages.length === 0 && (
                  <div className="msg-bubble bot-msg">
                    Namaste! 🙏 I am your Farmer Chatbot. Ask me anything about crops, weather, or farming!
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div key={i} className={`msg-bubble ${msg.sender === "You" ? "user-msg" : "bot-msg"}`}>
                    {msg.text === "Typing..." ? (
                      <div className="typing-indicator">
                        <span></span><span></span><span></span>
                      </div>
                    ) : (
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="chat-footer">
                <button
                  className={`btn btn-mic ${isRecording ? "recording" : ""}`}
                  onClick={handleVoiceSearch}
                  title="Speak your query"
                >
                  🎤
                </button>

                <input
                  type="text"
                  className="chat-input"
                  placeholder={isRecording ? "Listening..." : "Type or speak here..."}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                />

                <button className="btn btn-send" onClick={sendMessage}>
                  ➤
                </button>
              </div>

            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default App;