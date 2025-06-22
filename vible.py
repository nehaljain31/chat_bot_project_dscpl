import streamlit as st
import os
import json
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
import ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

import warnings
warnings.filterwarnings("ignore")

SOCIALVERSE_TOKEN = os.getenv("SOCIALVERSE_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



@dataclass
class UserProfile:
    name: str = ""
    preferred_topics: List[str] = None
    notification_times: Dict[str, str] = None
    spiritual_goals: List[str] = None
    prayer_requests: List[str] = None
    current_program: Optional[str] = None
    program_day: int = 0
    last_interaction: str = ""

    def __post_init__(self):
        if self.preferred_topics is None:
            self.preferred_topics = []
        if self.notification_times is None:
            self.notification_times = {}
        if self.spiritual_goals is None:
            self.spiritual_goals = []
        if self.prayer_requests is None:
            self.prayer_requests = []

class SpiritualKnowledgeBase:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            self.vectorstore = None
            self.initialize_knowledge_base()
        except Exception as e:
            st.error(f"Failed to initialize knowledge base: {str(e)}")
        
            self.embeddings = None
            self.vectorstore = None

    def initialize_knowledge_base(self):
        try:
            spiritual_content = [
                "When dealing with anxiety and worry, remember that God's peace surpasses all understanding. Cast your anxieties on Him because He cares for you.",
                "For those struggling with fear, remember that perfect love casts out fear. God has not given you a spirit of fear, but of power, love, and sound mind.",
                "When feeling overwhelmed, take heart in knowing that God will not give you more than you can handle. His grace is sufficient for you.",
                "For guidance in relationships, remember to love others as you love yourself. Forgiveness and understanding are key to healthy relationships.",
                "When seeking purpose, know that God has plans for you - plans to prosper you and not to harm you, to give you hope and a future.",
                "The ACTS model of prayer includes Adoration (praising God), Confession (acknowledging sins), Thanksgiving (expressing gratitude), and Supplication (making requests).",
                "Jesus taught us to pray in Matthew 6:9-13, giving us the Lord's Prayer as a model for our own prayers.",
                "Christian meditation involves focusing on God's word, His character, and His promises. It's about filling your mind with truth.",
                "Spiritual disciplines include prayer, Bible study, fasting, worship, service, and fellowship with other believers.",
                "For those battling addiction, remember that God offers freedom and healing. Seek support from your faith community and professional help.",
                "When facing depression, know that God is close to the brokenhearted. Professional help combined with spiritual support can bring healing.",
            ]

            if self.embeddings:
                documents = [Document(page_content=content) for content in spiritual_content]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = text_splitter.split_documents(documents)
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        except Exception as e:
            st.warning(f"Could not initialize vector store: {str(e)}")
            self.vectorstore = None

    def get_relevant_content(self, query: str, k: int = 3) -> List[str]:
        try:
            if self.vectorstore:
                docs = self.vectorstore.similarity_search(query, k=k)
                return [doc.page_content for doc in docs]
        except Exception as e:
            st.warning(f"Error retrieving relevant content: {str(e)}")
        
        
        spiritual_responses = {
            "anxiety": "When dealing with anxiety and worry, remember that God's peace surpasses all understanding.",
            "fear": "For those struggling with fear, remember that perfect love casts out fear.",
            "depression": "When facing depression, know that God is close to the brokenhearted.",
            "prayer": "Prayer is our direct line to God. The ACTS model can help structure your prayers.",
            "purpose": "When seeking purpose, know that God has plans for you - plans to prosper you."
        }
        
        query_lower = query.lower()
        for keyword, response in spiritual_responses.items():
            if keyword in query_lower:
                return [response]
        
        return ["God loves you and is always with you on your spiritual journey."]

class DSCPLChatbot:
    def __init__(self):
        try:
            self.model_name = "gemma3:1b"
        
        
            test_response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': 'test'}]
        )
        
        except Exception as e:
            st.error(f"Model {self.model_name} not available. Error: {str(e)}")
            st.stop()

        self.knowledge_base = SpiritualKnowledgeBase()
        self.user_profile = self.load_user_profile()
        self.conversation_history = []
        self.setup_system_prompt()

    def load_user_profile(self) -> UserProfile:
        try:
            if 'user_profile' in st.session_state:
                profile_dict = st.session_state.user_profile
                return UserProfile(**profile_dict)
        except Exception as e:
            st.warning(f"Could not load user profile: {str(e)}")
        return UserProfile()

    def save_user_profile(self):
        try:
            st.session_state.user_profile = asdict(self.user_profile)
        except Exception as e:
            st.warning(f"Could not save user profile: {str(e)}")

    def setup_system_prompt(self):
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are DSCPL, a warm, compassionate spiritual companion and mentor. You provide biblical guidance, prayer support, and encouragement to help people grow in their faith journey.

Your personality:
- Warm, caring, and non-judgmental
- Biblically grounded but not preachy
- Encouraging and hopeful
- A good listener who asks thoughtful questions
- Supportive of all denominations and spiritual backgrounds

Remember:
- The user's name is: {user_name}
- Their spiritual goals: {spiritual_goals}
- Current program: {current_program}
- Preferred topics: {preferred_topics}

Context from spiritual knowledge base:
{spiritual_context}

Always respond in a warm, human-like manner as if you're a trusted spiritual friend. Keep responses concise but meaningful."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    def get_response(self, user_input: str) -> str:
        try:
            spiritual_context = self.knowledge_base.get_relevant_content(user_input)
            context_text = "\n".join(spiritual_context) if spiritual_context else "No specific context found."

            
            system_content = f"""You are DSCPL, a warm, compassionate spiritual companion and mentor. You provide biblical guidance, prayer support, and encouragement to help people grow in their faith journey.

Your personality:
- Warm, caring, and non-judgmental
- Biblically grounded but not preachy
- Encouraging and hopeful
- A good listener who asks thoughtful questions
- Supportive of all denominations and spiritual backgrounds

Remember:
- The user's name is: {self.user_profile.name or "friend"}
- Their spiritual goals: {", ".join(self.user_profile.spiritual_goals) if self.user_profile.spiritual_goals else "not specified yet"}
- Current program: {self.user_profile.current_program or "none"}
- Preferred topics: {", ".join(self.user_profile.preferred_topics) if self.user_profile.preferred_topics else "not specified yet"}

Context from spiritual knowledge base:
{context_text}

Always respond in a warm, human-like manner as if you're a trusted spiritual friend. Keep responses concise but meaningful.

Recent conversation:
"""
            
            
            for msg in self.conversation_history[-6:]:  
                if isinstance(msg, HumanMessage):
                    system_content += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    system_content += f"DSCPL: {msg.content}\n"
            
            system_content += f"\nUser: {user_input}\nDSCPL:"

            try:
                response = ollama.chat(
                model=self.model_name,
                messages=[
            {
                'role': 'user',
                'content': system_content
            }
        ]
    )
                reply = response['message']['content'].strip()
            except Exception as e:
                reply = f"Error generating response: {str(e)}"

            
            self.conversation_history.append(HumanMessage(content=user_input))
            self.conversation_history.append(AIMessage(content=reply))
            
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            self.extract_user_info(user_input, reply)
            return reply
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble responding right now. Please try again in a moment."

    def extract_user_info(self, user_input: str, bot_response: str):
        try:
            user_lower = user_input.lower()
            
            # Extract name
            if not self.user_profile.name:
                name_patterns = ["my name is", "i'm", "i am", "call me"]
                for pattern in name_patterns:
                    if pattern in user_lower:
                        words = user_input.split()
                        pattern_words = pattern.split()
                        try:
                            pattern_index = next(i for i in range(len(words) - len(pattern_words) + 1) 
                                               if ' '.join(words[i:i+len(pattern_words)]).lower() == pattern)
                            if pattern_index + len(pattern_words) < len(words):
                                name = words[pattern_index + len(pattern_words)].strip(".,!?")
                                if name.isalpha() and len(name) > 1:
                                    self.user_profile.name = name.title()
                                    break
                        except (StopIteration, IndexError):
                            continue

        
            spiritual_keywords = {
                "anxiety": "anxiety and worry",
                "fear": "overcoming fear", 
                "depression": "dealing with depression",
                "relationship": "relationships",
                "healing": "healing",
                "purpose": "purpose and calling",
                "addiction": "addiction recovery",
                "prayer": "prayer life",
                "forgiveness": "forgiveness",
                "bible": "bible study",
                "worship": "worship",
                "faith": "faith building"
            }
            
            for keyword, topic in spiritual_keywords.items():
                if keyword in user_lower and topic not in self.user_profile.preferred_topics:
                    self.user_profile.preferred_topics.append(topic)

            
            time_keywords = ["morning", "evening", "night", "afternoon"]
            for time_word in time_keywords:
                if time_word in user_lower:
                    if any(word in user_lower for word in ["notification", "remind", "alert", "schedule"]):
                        self.user_profile.notification_times["devotion"] = time_word

            self.save_user_profile()
            
        except Exception as e:
            st.warning(f"Could not extract user info: {str(e)}")

def get_socialverse_content():
    """Fetch content from SocialVerse API with proper error handling"""
    if not SOCIALVERSE_TOKEN:
        return []
    
    url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=10"
    headers = {
        "Flic-Token": SOCIALVERSE_TOKEN,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("posts", [])
        else:
            st.warning(f"SocialVerse API returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch SocialVerse content: {str(e)}")
    except Exception as e:
        st.warning(f"Unexpected error fetching content: {str(e)}")
    
    return []

def main():
    st.set_page_config(
        page_title="DSCPL - Your Spiritual Companion", 
        page_icon="🙏", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E4057;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .user-message {
       background-color: #E3F2FD;
        color: #1A1A1A;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4A90E2;
    }
    .bot-message {
       background-color: #F5F5F5;
        color: #1A1A1A;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #7BC142;
    }
    .stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>🙏 DSCPL - Your Spiritual Companion</h1>", unsafe_allow_html=True)

    
    try:
        if 'chatbot' not in st.session_state:
            with st.spinner("Initializing your spiritual companion..."):
                st.session_state.chatbot = DSCPLChatbot()
        
        chatbot = st.session_state.chatbot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.stop()

    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        welcome_msg = "Hello! I'm DSCPL, your personal spiritual companion. I'm here to walk alongside you in your faith journey. What's on your heart today?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'><strong>DSCPL:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    
    if prompt := st.chat_input("Share what's on your heart..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("DSCPL is reflecting on your message..."):
            try:
                response = chatbot.get_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = "I apologize, but I'm having trouble responding right now. Please try again in a moment."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(f"Error: {str(e)}")
        
        st.rerun()

    
    with st.sidebar:
        st.header("🌟 Your Spiritual Journey")
        
        
        if chatbot.user_profile.name:
            st.success(f"Welcome back, {chatbot.user_profile.name}!")
        else:
            st.info("Feel free to share your name so I can personalize our conversations!")
        
        
        if chatbot.user_profile.preferred_topics:
            st.subheader("🎯 Your Focus Areas")
            for topic in chatbot.user_profile.preferred_topics:
                st.write(f"• {topic}")
        
        
        if chatbot.user_profile.notification_times:
            st.subheader("⏰ Your Reminders")
            for notification_type, time in chatbot.user_profile.notification_times.items():
                st.write(f"• {notification_type.title()}: {time}")
        
        st.divider()
        
    
        st.subheader("🚀 Quick Spiritual Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🙏 Prayer Request"):
                request = "I would like to share a prayer request with you."
                st.session_state.messages.append({"role": "user", "content": request})
                try:
                    response = chatbot.get_response(request)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                st.rerun()
        
        with col2:
            if st.button("📖 Daily Devotional"):
                devotional = "Can you share today's devotional message with me?"
                st.session_state.messages.append({"role": "user", "content": devotional})
                try:
                    response = chatbot.get_response(devotional)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                st.rerun()
        
        if st.button("🧘 Meditation Guide"):
            meditation = "I'd like some guidance for meditation and quiet time today."
            st.session_state.messages.append({"role": "user", "content": meditation})
            try:
                response = chatbot.get_response(meditation)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
            st.rerun()
        
        if st.button("✨ Spiritual Encouragement"):
            encouragement = "I could use some spiritual encouragement today."
            st.session_state.messages.append({"role": "user", "content": encouragement})
            try:
                response = chatbot.get_response(encouragement)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
            st.rerun()
        
        st.divider()
        
        
        st.subheader("🌐 Community Inspiration")
        
        try:
            posts = get_socialverse_content()
            if posts:
                for post in posts[:3]:
                    title = post.get("title", "Untitled")
                    title = title[:50] + "..." if len(title) > 50 else title
                    st.write(f"🎥 {title}")
            else:
                st.write("💫 Loading inspirational content...")
        except Exception as e:
            st.write("💫 Inspirational content will be available soon...")
        
        st.divider()
        
    
        if st.checkbox("Show Debug Info"):
            st.subheader("Debug Information")
            st.write(f"Profile: {chatbot.user_profile.name}")
            st.write(f"Topics: {len(chatbot.user_profile.preferred_topics)}")
            st.write(f"Messages: {len(st.session_state.messages)}")
            st.write(f"Gemini API: {'✓' if GEMINI_API_KEY else '✗'}")
            st.write(f"SocialVerse API: {'✓' if SOCIALVERSE_TOKEN else '✗'}")

if __name__ == "__main__":
    main()