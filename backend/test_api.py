#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://localhost:8000"

def test_init():
    print("=" * 60)
    print("Testing /api/init endpoint")
    print("=" * 60)
    
    payload = {
        "text": "hello world test",
        "config": {
            "n_vocab": 50257,
            "n_embd": 768,
            "n_layer": 2,
            "n_head": 12,
            "d_k": 64,
            "max_seq_len": 512
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/init", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Session ID: {data['session_id']}")
        print(f"✓ Tokens: {data['tokens']}")
        print(f"✓ Token texts: {data['token_texts']}")
        print(f"✓ Total steps: {data['total_steps']}")
        print(f"✓ Embeddings shape: {len(data['initial_state']['embeddings'])} x {len(data['initial_state']['embeddings'][0])}")
        return data['session_id'], data['total_steps']
    else:
        print(f"✗ Failed with status code: {response.status_code}")
        print(f"  Response: {response.text}")
        return None, None


def test_steps(session_id, total_steps):
    print("\n" + "=" * 60)
    print("Testing /api/step endpoint")
    print("=" * 60)
    
    test_steps = [0, 1, 5, 13, 14, 27]
    
    for step in test_steps:
        if step >= total_steps:
            continue
            
        payload = {
            "session_id": session_id,
            "step": step
        }
        
        response = requests.post(f"{BASE_URL}/api/step", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Step {step:2d}: Layer {data['layer_index']}, "
                  f"Type: {data['step_type']:20s}, "
                  f"Description: {data['description'][:40]}...")
        else:
            print(f"✗ Step {step} failed with status code: {response.status_code}")


def test_sessions():
    print("\n" + "=" * 60)
    print("Testing /api/sessions endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/sessions")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Total sessions: {data['total_sessions']}")
        print(f"✓ Session IDs: {data['session_ids'][:3]}...")
    else:
        print(f"✗ Failed with status code: {response.status_code}")


def test_delete_session(session_id):
    print("\n" + "=" * 60)
    print("Testing DELETE /api/session/{session_id}")
    print("=" * 60)
    
    response = requests.delete(f"{BASE_URL}/api/session/{session_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Deleted session: {data['session_id']}")
    else:
        print(f"✗ Failed with status code: {response.status_code}")


def main():
    print("Testing Transformer Simulator API")
    print("=" * 60)
    
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print(f"✓ Server is running at {BASE_URL}")
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return
    
    session_id, total_steps = test_init()
    
    if session_id:
        test_steps(session_id, total_steps)
        test_sessions()
        test_delete_session(session_id)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
