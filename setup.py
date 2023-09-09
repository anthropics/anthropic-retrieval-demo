from setuptools import setup, find_packages

VERSION = "0.1.1"
DESCRIPTION = "Claude Search and Retrieval Demo"
LONG_DESCRIPTION = "Lightweight wrapper around the Anthropic Python SDK to experiment with Claude's Search and Retrieval capabilities."

# Setting up
setup(
        name="claude_retriever", 
        version=VERSION,
        author="Anthropic",
        author_email="support@anthropic.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
    "aiohttp",
    "aiosignal",
    "anthropic",
    "anyio",
    "beautifulsoup4",
    "certifi", 
    "click",
    "elasticsearch",
    "filelock",
    "huggingface-hub",
    "idna",
    "joblib",
    "loguru",
    "mpmath",
    "multidict",
    "nest-asyncio",  
    "nltk",
    "numpy",
    "packaging",
    "pinecone-client",
    "psutil",
    "pydantic",
    "python-dateutil",
    "python-dotenv",
    "pyyaml",
    "requests",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "sentencepiece",
    "six",
    "soupsieve",
    "sympy",
    "tenacity",
    "torch",
    "tornado",
    "transformers",
    "urllib3",
    "wikipedia"
],        
        keywords=["python", "anthropic"],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Private :: Do Not Upload"
        ]
)

