#libraray.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path
import torch
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import requests