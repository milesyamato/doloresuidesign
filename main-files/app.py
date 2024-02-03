#app.py
from library import *

app = Flask(__name__,
            template_folder='dist',
            static_folder='dist/assets')
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app)

name = "meta-llama/Llama-2-70b-chat-hf"
auth_token = "ghgh"

# Define your User and Message models here
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Other user fields

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Other message fields

@app.before_first_request
def load_resources():
    global tokenizer, model, embeddings, service_context, query_engine, documents
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/',
                                                 use_auth_token=auth_token, torch_dtype=torch.float16)
                                                # rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)

    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe...
    <</SYS>>[/INST]
    """

    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")
    llm = HuggingFaceLLM(context_window=4096,
                         max_new_tokens=256,
                         system_prompt=system_prompt,
                         query_wrapper_prompt=query_wrapper_prompt,
                         model=model,
                         tokenizer=tokenizer)

    # Set up the embeddings and service context
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embeddings
    )
    set_global_service_context(service_context)

    # Load documents from the 'agent/data' directory
    documents = []
    data_directory = Path('D:\\UI build\\main-files\\agent\\temporal_dimension')  # Updated path to the directory containing your documents
    PyMuPDFReader = download_loader("PyMuPDFReader")
    loader = PyMuPDFReader()

    if data_directory.is_dir():
        for file_path in data_directory.iterdir():
            if file_path.is_file() and file_path.suffix == '.pdf':
                loaded_document = loader.load(file_path=file_path, metadata=True)
                documents.append(loaded_document)
    
    # Assuming VectorStoreIndex and the query engine can handle multiple documents
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/login', methods=['POST'])
def login():
    # Handle login

@app.route('/logout')
def logout():
    # Handle logout

@app.route('/query', methods=['POST'])
def query():
    prompt = request.form['prompt']
    response = query_engine.query(prompt)
    return jsonify(response)

@app.route('/response_object', methods=['POST'])
def response_object():
    prompt = request.form['prompt']
    response = query_engine.query(prompt)
    return jsonify(response)

@app.route('/source_text', methods=['POST'])
def source_text():
    prompt = request.form['prompt']
    response = query_engine.query(prompt)
    return jsonify(response.get_formatted_sources())

@app.route('/chat-archive')
def chat_archive():
    chat_archive_folder = 'D:\\UI build\\main-files\\agent\\chat_archive' # Replace with your actual chat-archive folder path
    # Assuming you want to send a list of filenames in the chat-archive directory
    files = [f.name for f in Path(chat_archive_folder).iterdir() if f.is_file()]
    return jsonify(files)

@app.route('/weather')
def weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "current": "temperature_2m,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
    }
    response = requests.get(url, params=params)
    weather_data = response.json()

    # Convert temperature from Celsius to Fahrenheit
    if 'temperature_2m' in weather_data['current']:
        celsius_temp = weather_data['current']['temperature_2m']
        fahrenheit_temp = (celsius_temp * 9/5) + 32
        weather_data['current']['temperature_2m'] = fahrenheit_temp

    return render_template('weather.html', weather_data=weather_data)


# Real-time messaging handlers
@socketio.on('connect')
def handle_connect():
    # Handle a new WebSocket connection

@socketio.on('disconnect')
def handle_disconnect():
    # Handle WebSocket disconnection

@socketio.on('send_message')
def handle_send_message(data):
    # Handle a new message
    # Store the message in the database
    # Emit the message to the recipient(s)
    emit('new_message', data, broadcast=True)

# ... [other WebSocket event handlers] ...

if __name__ == '__main__':
    db.create_all()  # Create database tables
    socketio.run(app, debug=True)
    app.run(debug=True)