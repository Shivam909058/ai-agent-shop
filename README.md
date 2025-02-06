# 🤖 AI Agent Shop

A powerful AI agent generator built on the Agno framework that lets you create custom AI agents for any use case. Powered by RAG (Retrieval Augmented Generation) and Claude 3.5, this tool makes creating specialized AI agents as simple as describing what you need.

## 🌟 Features

- **Instant Agent Generation**: Create specialized AI agents with just a text description
- **Production-Ready**: Generates complete, working implementations following best practices
- **Customizable**: Adjust parameters and configurations to suit your needs
- **Easy to Use**: Simple web interface built with Streamlit

## 🛠️ Installation

1. Clone the repository:
```
git clone https://github.com/shivam909058/ai-agent-shop.git
```

2. Navigate to the project directory:
```
cd AI_AUTOMATION
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
streamlit run main.py
```

5. .env file contains the API keys for the tools used in the project.
# api key for claude 3.5
```
CLAUDE_API_KEY=your_claude_api_key
```
# api key for openai
```
OPENAI_API_KEY=your_openai_api_key
```

## 📖 Usage

1. Open the web interface in your browser.
2. Enter your agent description and click "Generate".
3. Review the generated code and make any necessary adjustments.
4. Download the generated code and use it in your projects.

ai-agent-shop/
├── main.py # Main application file
├── data.txt # Agno documentation and examples
├── requirements.txt # Project dependencies
├── .env # Environment variables (not tracked)
└── data/ # Data directory
└── vectordb/ # Vector database (not tracked)

# use docker compose to run the application
```
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your_key -e OPENAI_API_KEY=your_key yourusername/agno-agent-generator:latest
```

2. Open your browser and visit: http://localhost:8501

Requirements:
- Docker installed on your machine
- OpenAI API key
- Anthropic API key

Note: Replace `your_openai_key` and `your_anthropic_key` with your actual API keys.

# Create .env file
echo "OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key" > .env

# Run with env file
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your_key -e OPENAI_API_KEY=your_key yourusername/agno-agent-generator:latest
