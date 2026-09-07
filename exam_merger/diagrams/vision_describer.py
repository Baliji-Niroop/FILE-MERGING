import os
import base64
import types

try:
    import anthropic
except ImportError:
    class _AnthropicStub:
        def __init__(self, *args, **kwargs):
            raise ImportError("anthropic package is not installed")

    anthropic = types.SimpleNamespace(Anthropic=_AnthropicStub)

def describe_image(image_bytes: bytes, media_type: str = "image/png") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY environment variable is not set. Skipping image description.")
        return ""
        
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Convert image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Claude vision request
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this technical/educational diagram in 2-3 sentences. State what it depicts, name any labeled components visible, and explain what concept or process it illustrates. Be factual and concise — no creative language."
                        }
                    ],
                }
            ],
        )
        
        if message.content and len(message.content) > 0:
            return message.content[0].text.strip()
    except Exception as e:
        print(f"Error describing image via Claude Vision API: {e}")
        
    return ""
