# FastAPI OpenAI Integration

A FastAPI-based REST API that provides text generation and sentiment analysis capabilities using OpenAI's GPT models.

## Features

- Text generation using GPT-3.5-turbo
- Sentiment analysis of text input
- Rate limiting (5 requests per minute)
- Error handling for OpenAI API interactions
- Environment-based configuration
- Structured logging
- Docker support for easy deployment

## Project Structure

```
src/
├── api/
│   └── apis.py                 # API route definitions
├── exceptions/
│   └── openai.py              # OpenAI-specific error handling
├── schema/
│   ├── requests.py            # Request models
│   └── responses.py           # Response models
├── services/
│   └── openai_service.py      # OpenAI service integration
├── docker-compose.yml         # Docker compose configuration
├── Dockerfile                 # Docker build instructions
├── main.py                    # Application entry point
└── requirements.txt           # Project dependencies
```

## Prerequisites

- Docker and Docker Compose
- OpenAI API key

## Running with Docker Compose

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

3. Build and run the containers:
```bash
# Build and start the services in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the services
docker compose down
```

The API will be available at `http://localhost:8000`.

You can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Docker Compose Commands

Common commands for managing the application:

```bash
# Rebuild the containers
docker compose build

# Start the services
docker compose up

# Start specific service
docker compose up web

# Stop the services
docker compose down

# View service logs
docker compose logs

# View logs for specific service
docker compose logs web

# Check service status
docker compose ps

# Restart services
docker compose restart
```

### Troubleshooting Docker

If you encounter issues:

1. Check if the containers are running:
```bash
docker compose ps
```

2. Check the logs for errors:
```bash
docker compose logs web
```

3. Rebuild the containers:
```bash
docker compose down
docker compose up --build
```

## API Endpoints

### Generate Text

```http
POST /generate-text
```

Request body:
```json
{
    "prompt": "Your text prompt here",
    "max_tokens": 100
}
```

### Analyze Sentiment

```http
POST /analyze-sentiment
```

Request body:
```json
{
    "text": "Text to analyze sentiment"
}
```

Response format for both endpoints:
```json
{
    "result": "Generated text or sentiment analysis result"
}
```

## Rate Limiting

Both endpoints are rate-limited to 5 requests per minute per IP address.

## Error Handling

The API includes comprehensive error handling for:
- OpenAI API errors
- Rate limiting
- Invalid requests
- Server errors

## Development

### Adding New Endpoints

1. Define request/response models in `schema/`
2. Add service methods in `services/`
3. Create new routes in `api/apis.py`

### Error Handling

Custom exception handling is implemented in `exceptions/openai.py`. Add new exception handlers as needed.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

