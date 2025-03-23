from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import requests
import json
import re
import os
import logging

app = FastAPI()

# Add CORS support for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LLAMA_CPP_URL = "http://localhost:8083/completion"  # Your llama.cpp server URL
CACHE_DIR = "./prompt_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Models for request/response structure
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# Function to detect and parse tool calls from model output
def detect_tool_calls(text: str, available_tools: List[Dict]) -> List[ToolCall]:
    tool_calls = []
    
    # Patterns for tool detection - adjust based on your model's output format
    # This is a simplified example - you'll need to adapt this for your specific model
    function_pattern = r'\{\s*"name":\s*"([^"]+)"[^}]*"arguments":\s*(\{[^}]+\})\s*\}'
    
    for match in re.finditer(function_pattern, text):
        tool_name = match.group(1)
        args_str = match.group(2)
        
        # Find matching tool from available tools
        matching_tool = next((t for t in available_tools if t["function"]["name"] == tool_name), None)
        
        if matching_tool:
            try:
                args = json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{len(tool_calls)}",
                        function={
                            "name": tool_name,
                            "arguments": args_str
                        }
                    )
                )
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool arguments: {args_str}")
    
    return tool_calls

# Format messages for llama.cpp
def format_messages_for_llama(messages: List[Message], tools: Optional[List[Dict]] = None) -> str:
    prompt = ""
    
    # Add system message with tool definitions if tools are provided
    if tools:
        tools_json = json.dumps(tools, indent=2)
        system_prompt = f"You are Claude, an AI assistant by Anthropic. You have access to the following tools:\n\n{tools_json}\n\n"
        system_prompt += "When you need to use a tool, respond with a JSON object with the tool name and arguments."
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    # Add all messages
    for msg in messages:
        role = msg.role
        
        # Handle content formatting
        if isinstance(msg.content, str):
            content = msg.content
        else:
            # Handle complex content structure (like with image URLs)
            content_parts = []
            for part in msg.content:
                if part.get("type") == "text":
                    content_parts.append(part.get("text", ""))
                # Handle other content types as needed
            content = "\n".join(content_parts)
        
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add assistant prefix to get the model to respond
    prompt += "<|im_start|>assistant\n"
    
    return prompt

# Cache for storing and retrieving conversation history
class PromptCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, conversation_id: str) -> str:
        return os.path.join(self.cache_dir, f"{conversation_id}.json")
    
    def save_conversation(self, conversation_id: str, messages: List[Dict]) -> None:
        with open(self.get_cache_path(conversation_id), 'w') as f:
            json.dump(messages, f)
    
    def load_conversation(self, conversation_id: str) -> List[Dict]:
        try:
            with open(self.get_cache_path(conversation_id), 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

# Context window management
def manage_context_window(
    messages: List[Message], 
    max_tokens: int = 8000,
    strategy: str = "truncate"  # Options: truncate, middle-out, summarize
) -> List[Message]:
    # Simple token estimation - replace with a proper tokenizer
    def estimate_tokens(text: str) -> int:
        return len(text.split())
    
    if strategy == "truncate":
        # Keep the most recent messages that fit in the context
        total_tokens = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_content = messages[i].content
            if isinstance(msg_content, list):
                # Handle content list format
                msg_text = " ".join([p.get("text", "") for p in msg_content if p.get("type") == "text"])
            else:
                msg_text = msg_content
            
            tokens = estimate_tokens(msg_text)
            if total_tokens + tokens > max_tokens:
                return messages[i+1:]
            total_tokens += tokens
        
        return messages
    
    elif strategy == "middle-out":
        # Keep first few and last few messages, remove middle
        if len(messages) < 4:
            return messages
            
        # Estimate tokens for all messages
        token_counts = []
        for msg in messages:
            msg_content = msg.content
            if isinstance(msg_content, list):
                msg_text = " ".join([p.get("text", "") for p in msg_content if p.get("type") == "text"])
            else:
                msg_text = msg_content
            token_counts.append(estimate_tokens(msg_text))
        
        total_tokens = sum(token_counts)
        if total_tokens <= max_tokens:
            return messages
            
        # Always keep the system message if present
        start_idx = 1 if messages[0].role == "system" else 0
        
        # Always keep at least the last 2 messages
        must_keep = token_counts[start_idx] + sum(token_counts[-2:])
        remaining_budget = max_tokens - must_keep
        
        # Remove messages from the middle until we fit
        middle_start = start_idx + 1
        middle_end = len(messages) - 2
        
        if middle_start >= middle_end:
            # Not enough messages to remove from middle
            return messages
            
        # Create a new list with the kept messages
        kept_messages = [messages[start_idx]]
        
        # Add a summary message to replace removed content
        summary_msg = Message(
            role="system",
            content="[Some earlier messages have been summarized to fit within the context window]"
        )
        kept_messages.append(summary_msg)
        
        # Add the last two messages
        kept_messages.extend(messages[-2:])
        
        return kept_messages
    
    elif strategy == "summarize":
        # This would require an additional call to the model to summarize
        # For now, fall back to truncation
        logger.warning("Summarize strategy not fully implemented, falling back to truncation")
        return manage_context_window(messages, max_tokens, "truncate")
    
    return messages

@app.post("/v1/messages")
async def create_messages(request: CompletionRequest):
    """Handle chat completion requests in Claude API format"""
    try:
        # Manage context window based on a strategy
        managed_messages = manage_context_window(
            request.messages, 
            max_tokens=8000,  # Adjust based on your model's capabilities
            strategy="middle-out"
        )
        
        # Format messages for llama.cpp
        prompt = format_messages_for_llama(managed_messages, request.tools)
        
        # Prepare llama.cpp request
        llama_request = {
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop": ["<|im_end|>"],
            "stream": False
        }
        
        # Make request to llama.cpp server
        response = requests.post(LLAMA_CPP_URL, json=llama_request)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error from LLM server")
        
        llama_response = response.json()
        model_output = llama_response.get("content", "")
        
        # Process the model output
        tool_calls = []
        regular_response = model_output
        
        # Check if the response contains tool calls
        if request.tools:
            tool_calls = detect_tool_calls(model_output, request.tools)
            
            # If tool calls were detected, modify the response
            if tool_calls:
                # Remove the tool calls JSON from the regular response
                # This is a simplified approach - you might need more sophisticated parsing
                for call in tool_calls:
                    call_json = json.dumps({"name": call.function["name"], "arguments": call.function["arguments"]})
                    regular_response = regular_response.replace(call_json, "")
        
        # Format response according to OpenAI/Claude format
        response_object = {
            "id": f"chatcmpl-{os.urandom(4).hex()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": regular_response.strip() if not tool_calls else None,
                        "tool_calls": [call.dict() for call in tool_calls] if tool_calls else None
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(model_output.split()),
                "total_tokens": len(prompt.split()) + len(model_output.split())
            }
        }
        
        return response_object
        
    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e))
    

    import asyncio
from fastapi.responses import StreamingResponse

@app.post("/v1/messages:stream")
async def stream_messages(request: CompletionRequest):
    """Handle streaming chat completion requests in Claude API format"""
    async def generate():
        try:
            # Apply same context management as non-streaming version
            managed_messages = manage_context_window(
                request.messages, 
                max_tokens=8000,
                strategy="middle-out"
            )
            
            prompt = format_messages_for_llama(managed_messages, request.tools)
            
            # Request to llama.cpp with streaming enabled
            llama_request = {
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stop": ["<|im_end|>"],
                "stream": True
            }
            
            # Make request to llama.cpp streaming endpoint
            async with httpx.AsyncClient() as client:
                async with client.stream('POST', LLAMA_CPP_URL, json=llama_request, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_detail = await response.read()
                        yield f"data: {json.dumps({'error': f'Error from LLM server: {error_detail}'})}\n\n"
                        return
                    
                    buffer = ""
                    message_id = f"chatcmpl-{os.urandom(4).hex()}"
                    
                    async for chunk in response.aiter_text():
                        try:
                            chunk_data = json.loads(chunk)
                            token = chunk_data.get("content", "")
                            buffer += token
                            
                            # Check for potential tool calls in accumulating buffer
                            tool_calls = []
                            if request.tools:
                                tool_calls = detect_tool_calls(buffer, request.tools)
                            
                            # Format streaming response like Claude
                            stream_resp = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": token if not tool_calls else None,
                                            "tool_calls": [call.dict() for call in tool_calls] if tool_calls else None
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            
                            yield f"data: {json.dumps(stream_resp)}\n\n"
                            
                        except json.JSONDecodeError:
                            # Handle non-JSON chunks if any
                            continue
                    
                    # Send the final chunk with finish_reason
                    finish_reason = "tool_calls" if tool_calls else "stop"
                    final_resp = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_resp)}\n\n"
                    yield "data: [DONE]\n\n"
                    
        except Exception as e:
            logger.exception("Error in streaming")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

import uvicorn
import time
import httpx

if __name__ == "__main__":
    logger.info("Starting Claude API middleware for llama.cpp")
    logger.info(f"Connecting to llama.cpp server at: {LLAMA_CPP_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
