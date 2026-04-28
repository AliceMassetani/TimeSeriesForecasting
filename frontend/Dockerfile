# Use official Node.js image as base
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Install Angular CLI globally
RUN npm install -g @angular/cli

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Expose the port Angular runs on
EXPOSE 4200

# Start the application in development mode
CMD ["ng", "serve", "--host", "0.0.0.0", "--poll", "2000"]
