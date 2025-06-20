import streamlit as st
import os
import json
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import schedule
import time
import threading
from dotenv import load_dotenv
load_dotenv()

# Optional (if you want to replace hardcoded token later)
SOCIALVERSE_TOKEN = os.getenv("SOCIALVERSE_TOKEN")
# Configuration
OLLAMA_MODEL = "llama3.1:8b"  # Local LLaMA model
EMBEDDING_MODEL = "nomic-embed-text"

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
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = None
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize the RAG knowledge base with spiritual content"""
        spiritual_content = [
            # Bible verses for different situations
            "When dealing with anxiety and worry, remember Philippians 4:6-7: 'Do not be anxious about anything, but in every situation, by prayer and petition, with thanksgiving, present your requests to God. And the peace of God, which transcends all understanding, will guard your hearts and your minds in Christ Jesus.'",
            
            "For those struggling with fear, Isaiah 41:10 provides comfort: 'So do not fear, for I am with you; do not be dismayed, for I am your God. I will strengthen you and help you; I will uphold you with my righteous right hand.'",
            
            "When feeling overwhelmed, cast your burdens on the Lord as stated in Psalm 55:22: 'Cast your cares on the Lord and he will sustain you; he will never let the righteous be shaken.'",
            
            "For guidance in relationships, Ephesians 4:32 teaches: 'Be kind and compassionate to one another, forgiving each other, just as in Christ God forgave you.'",
            
            "When seeking purpose, Jeremiah 29:11 reminds us: 'For I know the plans I have for you, declares the Lord, plans to prosper you and not to harm you, to give you hope and a future.'",
            
            # Prayer guidance
            "The ACTS model of prayer includes: Adoration (praising God), Confession (acknowledging our sins), Thanksgiving (expressing gratitude), and Supplication (making requests).",
            
            "Jesus taught us to pray in Matthew 6:9-13 with the Lord's Prayer, which covers worship, daily needs, forgiveness, and protection from evil.",
            
            # Meditation and spiritual practices
            "Christian meditation involves focusing on God's word and character. Psalm 1:2 speaks of delighting in the law of the Lord and meditating on it day and night.",
            
            "Spiritual disciplines include prayer, Bible study, worship, fellowship, service, and fasting. These practices help us grow closer to God.",
            
            # Overcoming struggles
            "For those battling addiction, 1 Corinthians 10:13 offers hope: 'No temptation has overtaken you except what is common to mankind. And God is faithful; he will not let you be tempted beyond what you can bear.'",
            
            "When facing depression, remember that God is close to the brokenhearted (Psalm 34:18) and that weeping may endure for a night, but joy comes in the morning (Psalm 30:5).",
        ]
        
        # Create documents
        documents = [Document(page_content=content) for content in spiritual_content]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
    
    def get_relevant_content(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant spiritual content based on user query"""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        return []

class DSCPLChatbot:
    def __init__(self):
        self.llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)
        self.knowledge_base = SpiritualKnowledgeBase()
        self.user_profile = self.load_user_profile()
        self.conversation_history = []
        self.setup_system_prompt()
    
    def load_user_profile(self) -> UserProfile:
        """Load user profile from session state or create new one"""
        if 'user_profile' in st.session_state:
            profile_dict = st.session_state.user_profile
            return UserProfile(**profile_dict)
        return UserProfile()
    
    def save_user_profile(self):
        """Save user profile to session state"""
        st.session_state.user_profile = asdict(self.user_profile)
    
    def setup_system_prompt(self):
        """Setup the system prompt for the chatbot"""
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are DSCPL, a warm, compassionate spiritual companion and mentor. Your purpose is to guide people in their faith journey through personalized devotionals, prayers, meditation, and accountability support.

Your personality traits:
- Speak like a caring friend, not a formal assistant
- Be empathetic and understanding
- Use natural, conversational language
- Ask thoughtful follow-up questions
- Remember details about the user's spiritual journey
- Offer practical, biblical guidance
- Be encouraging and supportive

Key capabilities:
- Provide personalized daily devotionals
- Guide users through prayer using the ACTS model
- Lead meditation sessions with scripture
- Offer accountability support for various struggles
- Schedule and remind about spiritual practices
- Share relevant Bible verses and spiritual wisdom

Remember:
- The user's name is: {user_name}
- Their spiritual goals: {spiritual_goals}
- Current program: {current_program}
- Preferred topics: {preferred_topics}

Context from spiritual knowledge base:
{spiritual_context}

Always respond in a warm, human-like manner as if you're a trusted spiritual friend."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    
    def get_response(self, user_input: str) -> str:
        """Generate response using RAG and conversation context"""
        # Get relevant spiritual content
        spiritual_context = self.knowledge_base.get_relevant_content(user_input)
        context_text = "\n".join(spiritual_context) if spiritual_context else "No specific context found."
        
        # Prepare the prompt
        prompt = self.system_prompt.format_messages(
            user_name=self.user_profile.name or "friend",
            spiritual_goals=", ".join(self.user_profile.spiritual_goals) if self.user_profile.spiritual_goals else "not specified yet",
            current_program=self.user_profile.current_program or "none",
            preferred_topics=", ".join(self.user_profile.preferred_topics) if self.user_profile.preferred_topics else "not specified yet",
            spiritual_context=context_text,
            chat_history=self.conversation_history[-10:],  # Keep last 10 messages
            input=user_input
        )
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=response))
        
        # Extract and update user information from the conversation
        self.extract_user_info(user_input, response)
        
        return response
    
    def extract_user_info(self, user_input: str, bot_response: str):
        """Extract and update user information from conversation"""
        user_lower = user_input.lower()
        
        # Extract name
        if not self.user_profile.name:
            if "my name is" in user_lower or "i'm" in user_lower or "i am" in user_lower:
                words = user_input.split()
                for i, word in enumerate(words):
                    if word.lower() in ["is", "i'm", "am"] and i + 1 < len(words):
                        potential_name = words[i + 1].strip(".,!?")
                        if potential_name.isalpha():
                            self.user_profile.name = potential_name.title()
                            break
        
        # Extract spiritual topics/struggles
        spiritual_keywords = {
            "anxiety": "anxiety and worry",
            "fear": "overcoming fear",
            "depression": "dealing with depression",
            "relationship": "relationships",
            "healing": "healing",
            "purpose": "purpose and calling",
            "addiction": "addiction recovery",
            "prayer": "prayer life",
            "forgiveness": "forgiveness"
        }
        
        for keyword, topic in spiritual_keywords.items():
            if keyword in user_lower and topic not in self.user_profile.preferred_topics:
                self.user_profile.preferred_topics.append(topic)
        
        # Extract time preferences for notifications
        time_keywords = ["morning", "evening", "night", "afternoon", "dawn", "dusk"]
        for time_word in time_keywords:
            if time_word in user_lower:
                if "notification" in user_lower or "remind" in user_lower or "devotion" in user_lower:
                    self.user_profile.notification_times["devotion"] = time_word
        
        self.save_user_profile()
    
    def schedule_notification(self, notification_type: str, time_preference: str):
        """Schedule notifications based on user preferences"""
        time_mapping = {
            "morning": "07:00",
            "afternoon": "12:00",
            "evening": "18:00",
            "night": "21:00"
        }
        
        scheduled_time = time_mapping.get(time_preference.lower(), "07:00")
        
        # In a real implementation, you would integrate with a notification service
        # For now, we'll just store the preference
        self.user_profile.notification_times[notification_type] = scheduled_time
        self.save_user_profile()
        
        return f"Great! I've scheduled your {notification_type} notifications for {scheduled_time} ({time_preference}). You'll receive gentle reminders to connect with God during your preferred time."

def get_socialverse_content():
    """Fetch content from SocialVerse API"""
    url = "https://api.socialverseapp.com/posts/summary/get?page=1&page_size=10"
    headers = {
        "Flic-Token": "flic_b1c6b09d98e2d4884f61b9b3131dbb27a6af84788e4a25db067a22008ea9cce5",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("posts", [])
    except requests.exceptions.RequestException:
        pass
    
    return []

def main():
    st.set_page_config(
        page_title="DSCPL - Your Spiritual Companion",
        page_icon="🙏",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #4A90E2;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .user-message {
        background-color: #E3F2FD;
        color: #1A1A1A;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
    }
    .bot-message {
        background-color: #F5F5F5;
        color: #1A1A1A;
        padding: 1em;
        border-radius: 10px;
        margin: 0.5em 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DSCPLChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Header
    st.markdown('<h1 class="main-header">🙏 DSCPL</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Personal Spiritual Companion</p>', unsafe_allow_html=True)
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        welcome_msg = "Hello! I'm DSCPL, your personal spiritual companion. I'm here to walk with you on your faith journey - whether you need daily devotionals, prayer guidance, meditation support, or just someone to talk to about your spiritual life. What's on your heart today?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message"><strong>DSCPL:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Share what's on your heart..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        with st.spinner("DSCPL is reflecting on your message..."):
            response = chatbot.get_response(prompt)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display new messages
        st.rerun()
    
    # Sidebar with user profile and features
    with st.sidebar:
        st.header("Your Spiritual Journey")
        
        if chatbot.user_profile.name:
            st.write(f"Welcome back, {chatbot.user_profile.name}! 🌟")
        
        if chatbot.user_profile.preferred_topics:
            st.subheader("Your Focus Areas")
            for topic in chatbot.user_profile.preferred_topics:
                st.write(f"• {topic}")
        
        if chatbot.user_profile.notification_times:
            st.subheader("Your Reminders")
            for notification_type, time in chatbot.user_profile.notification_times.items():
                st.write(f"• {notification_type.title()}: {time}")
        
        # Quick actions
        st.subheader("Quick Spiritual Actions")
        if st.button("🙏 Request Prayer"):
            prayer_request = "I would like to share a prayer request with you."
            st.session_state.messages.append({"role": "user", "content": prayer_request})
            response = chatbot.get_response(prayer_request)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("📖 Daily Devotional"):
            devotional_request = "Can you share today's devotional with me?"
            st.session_state.messages.append({"role": "user", "content": devotional_request})
            response = chatbot.get_response(devotional_request)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("🧘 Meditation Guide"):
            meditation_request = "I'd like some guidance for meditation today."
            st.session_state.messages.append({"role": "user", "content": meditation_request})
            response = chatbot.get_response(meditation_request)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Inspirational content
        st.subheader("Community Inspiration")
        posts = get_socialverse_content()
        if posts:
            for post in posts[:3]:  # Show only first 3 posts
                title = post.get("title", "Untitled")
                if len(title) > 50:
                    title = title[:50] + "..."
                st.write(f"🎥 {title}")
        else:
            st.write("Loading inspirational content...")

if __name__ == "__main__":
    main()