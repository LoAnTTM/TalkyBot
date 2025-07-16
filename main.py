import threading
import queue
import time

from thread.thread_wakeup import WakeWordThread    # Thread 1
from thread.thread_stt import STTConversationThread  # Thread 2
from thread.thread_tts import TTSThread           # Thread 3
from thread.thread_vad import VADThread           # Voice Activity Detector
from components.vad import VoiceActivityDetector
from audio.mic_stream import AudioStream

def main():
    # 1. Tạo các queue để truyền dữ liệu giữa các thread
    text_to_tts_queue = queue.Queue()
    audio_to_stt_queue = queue.Queue()

    # 2. Khởi tạo VAD và AudioStream (chỉ 1 luồng âm thanh)
    vad = VoiceActivityDetector()
    audio_stream = AudioStream()

    # 3. Khởi tạo các thread
    wakeup_thread = WakeWordThread()
    
    # Thread 3: TTS nhận câu trả lời để nói
    tts_thread = TTSThread()
    tts_thread.start()

    # Thread 2: STT + Chatbot nhận audio từ VAD qua queue
    def on_final_result(text):
        print(f"[Main] STT final result: {text}")
        # Gửi câu trả lời Chatbot vào queue của TTS
        # Lấy phản hồi chatbot
        response = stt_thread.chatbot.get_response(text)
        print(f"[Main] Chatbot response: {response}")
        text_to_tts_queue.put(response)

    stt_thread = STTConversationThread(audio_queue=audio_to_stt_queue, response_callback=on_final_result)
    
    # Callback để theo dõi trạng thái VAD
    def vad_status_callback(info):
        if info['is_speaking']:
            print(f"🎤 Speech detected - Duration: {info['duration']:.1f}s")
        else:
            print("🔇 Speech ended")
    
    # VAD Thread - luồng duy nhất nghe microphone và gửi audio cho STT
    vad_thread = VADThread(vad, audio_stream, audio_queue=audio_to_stt_queue, status_callback=vad_status_callback)
    
    # Truyền queue cho TTS thread
    tts_thread.text_queue = text_to_tts_queue

    # 4. Start các thread
    wakeup_thread.start()
    vad_thread.start()
    stt_thread.start()

    print("System started. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
            # TODO: Kiểm soát logic vòng hội thoại dựa vào vad_thread, ví dụ reset khi 15s ko có voice
    except KeyboardInterrupt:
        print("Stopping all threads...")

        # Stop threads that have stop() method
        vad_thread.stop()
        stt_thread.stop()
        tts_thread.stop()

        # WakeWordThread is daemon, will stop automatically
        vad_thread.join()
        stt_thread.join()
        tts_thread.join()

    print("System stopped cleanly.")


if __name__ == "__main__":
    main()
