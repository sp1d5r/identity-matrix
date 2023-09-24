# Services

This directory contains service modules, which are responsible for executing the main logic of the application, such as interacting with external APIs, handling database operations, and running complex algorithms. Each service module should be focused on a specific domain of the application.

## Job States:

```python
STARTED = "Video and Images have been uploaded"
PREPROCESSING = "Extracting the frames out"
FACE_SWAPPING = "Swapping faces"
CREATING_VIDEO = "Combine frames back"
FINISHED = "Complete"
```


## Current Services

firebase_service.py
This service is responsible for interacting with Firebase. It handles tasks such as uploading files to Firebase Storage.

## Adding New Services

When adding a new service, create a new Python file in this directory. The name of the file should clearly reflect its purpose (e.g., database_service.py for a service that interacts with a database).

Each service should expose functions that perform tasks related to a specific domain of the application. For example, a database_service.py might expose functions like get_user(id) or save_user(user).

Avoid putting any Flask-specific code in your service modules. They should not be aware of the HTTP request/response cycle. Instead, they should focus on the business logic of your application. This makes them easier to test and reuse.

## Testing

Each service should have corresponding tests, which can be placed in a tests directory. The tests should cover all the main functionality of the service and should run independently of the other parts of the application.

This is a basic outline and can be expanded upon as your project grows. The key is to keep each service focused on a specific area of functionality, making your codebase easier to understand and maintain.