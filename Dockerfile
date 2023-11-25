# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app3
WORKDIR /app3

# Upgrade pip and install dependencies
RUN pip3 install --upgrade numpy
RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install Django==3.2
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install joblib
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install scikit-learn
RUN pip3 install django djangorestframework
RUN pip3 install mlflow


# Copy the entire project directory into the container
COPY . /app3/

# Run the application
CMD tail -f/dev/null
