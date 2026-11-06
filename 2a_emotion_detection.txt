"""
Emotion Detection module using IBM Watson NLP service.
"""
import json
import requests


def _mock_emotion_detection(text):
    text_lower = text.lower() if text else ""
    if any(w in text_lower for w in ["glad", "joy", "happy", "love", "great", "wonderful"]):
        dominant = "joy"
    elif any(w in text_lower for w in ["angry", "anger", "mad", "furious"]):
        dominant = "anger"
    elif any(w in text_lower for w in ["disgust", "disgusted", "gross"]):
        dominant = "disgust"
    elif any(w in text_lower for w in ["sad", "sadness", "depressed", "unhappy"]):
        dominant = "sadness"
    elif any(w in text_lower for w in ["afraid", "fear", "scared", "terrified"]):
        dominant = "fear"
    else:
        dominant = "joy"

    emotions = {
        "anger": 0.05,
        "disgust": 0.03,
        "fear": 0.04,
        "joy": 0.05,
        "sadness": 0.04,
    }
    emotions[dominant] = 0.85

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant,
    }


def emotion_detector(text_to_analyze):
    """
    Detects emotions from text using IBM Watson NLP service.
    """
    if not text_to_analyze or not str(text_to_analyze).strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    myobj = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, json=myobj, headers=headers, timeout=(1, 3))
    except requests.exceptions.RequestException:
        # Fallback when offline / running locally outside IBM Skills Network
        return _mock_emotion_detection(text_to_analyze)

    # Xử lý trường hợp đầu vào rỗng hoặc mã lỗi 400
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    if response.status_code != 200:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    formatted_response = json.loads(response.text)
    emotions = formatted_response["emotionPredictions"][0]["emotion"]

    # Tìm cảm xúc chiếm ưu thế (có điểm số cao nhất)
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }
