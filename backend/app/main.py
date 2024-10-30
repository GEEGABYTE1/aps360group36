import asyncio
import json
from datetime import timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from data_file import data_store
from html_generator import generate_html
from auth import authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from dummy_pi import get_timestamp, generate_sensor_data, handle_comand

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # To allow all --> allow_origins=["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to get token from WebSocket query parameters
async def get_token_from_websocket(websocket: WebSocket):
    try:
        token = websocket.query_params["token"]
        return token
    except KeyError:
        raise WebSocketDisconnect(code=1008)

@app.get("/")
async def get():
    return HTMLResponse(generate_html(data_store))

@app.get("/entry")                 # Temporary route for testing for frontend and backend integration
async def websocket_test_entrpoint():
    
    return {"message": "Hello World from Backend!"}
        
@app.get("/ws_front")
async def get_actuator_data():
    try:
        while True:
            await generate_sensor_data()
            await get_timestamp()
            data = {
                "sensors": data_store["sensors"],
                "actuators": data_store["actuators"],
                "timestamp":data_store["timestamp"]
            }
            return data 
    except:
        return {} 

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Accept the WebSocket connection
    await websocket.accept()

    # Get token from the query parameters
    token = await get_token_from_websocket(websocket)
    user = await get_current_user(token)
    try:
        while True:
            await generate_sensor_data()
            await get_timestamp()
            # Send the current sensor data and actuator status
            data = {
                "sensors": data_store["sensors"],
                "actuators": data_store["actuators"],
                "timestamp":data_store["timestamp"]
            }
            await websocket.send_text(json.dumps(data))

            # Check if any commands are received
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                command = json.loads(message)

                # Handle actuator toggling
                await handle_comand(command)

            except asyncio.TimeoutError:
                # Continue sending sensor updates if no command is received
                pass

            # Wait for 1 second before sending the next update
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"Client {user['username']} disconnected")

# Token route for login
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}