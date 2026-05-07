import asyncio
import websockets
import json
import time
from .liveness import get_challenge_sequence, detect_liveness_video, DISPLAY_TEXT

ACTION_TIMEOUT_SEC = 10
FRAMES_REQUIRED = 15

async def handler(websocket):   # 只接收一个参数
    sequence = get_challenge_sequence(3)
    await websocket.send(json.dumps({
        "type": "challenge_start",
        "sequence": sequence,
        "display": [DISPLAY_TEXT.get(a, a) for a in sequence]
    }))
    
    for action in sequence:
        await websocket.send(json.dumps({
            "type": "next_action",
            "action": action,
            "display": DISPLAY_TEXT.get(action, action)
        }))
        
        frames = []
        start_time = time.time()
        while time.time() - start_time < ACTION_TIMEOUT_SEC:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                data = json.loads(message)
                if data.get("type") == "frame":
                    frames.append(data["data"])
                    if len(frames) >= FRAMES_REQUIRED:
                        break
            except asyncio.TimeoutError:
                continue
        
        result = detect_liveness_video(frames, required_action=action)
        if not result.get("action_completed", False):
            await websocket.send(json.dumps({
                "type": "liveness_failed",
                "reason": result.get("reason", "动作未完成或超时")
            }))
            return
        
        await websocket.send(json.dumps({"type": "action_ok"}))
    
    await websocket.send(json.dumps({"type": "liveness_passed"}))

async def start_server():
    async with websockets.serve(handler, "localhost", 8765):
        print("WebSocket活体检测服务运行在 ws://localhost:8765")
        await asyncio.Future()

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_server())