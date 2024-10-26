FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

# Copy the rest of your application
COPY app.py ./
COPY predict.py ./
COPY predict_client.py ./
COPY dv.bin ./
COPY model1.bin ./

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the app
CMD ["pipenv", "run", "python", "app.py"]
