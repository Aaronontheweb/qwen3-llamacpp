#!/bin/bash
# Start script with process manager for proper cleanup

echo "🚀 Starting Qwen3 server with process manager..."

# Trap signals to ensure cleanup
trap 'echo "Caught signal, cleaning up..."; kill $PID 2>/dev/null; exit' SIGINT SIGTERM

# Start the server
python openai_server.py "$@" &
PID=$!

echo "Server started with PID $PID"
echo "Press Ctrl+C to stop the server gracefully"

# Wait for the server process
wait $PID
EXIT_CODE=$?

echo "Server exited with code $EXIT_CODE"
exit $EXIT_CODE