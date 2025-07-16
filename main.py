import threading
import queue
import time

from thread_wakeup import WakeupThread    # Thread 1
from thread_stt import STTConversationThread  # Thread 2
from thread_tts import TTSThread           # Thread 3
from thread_vad import VADThread           # Voice Activity Detector

def main():
    # 1. Tạo các queue để truyền dữ liệu giữa các thread
    text_to_tts_queue = queue.Queue()

    # 2. Khởi tạo các thread
    wakeup_thread = WakeupThread()
    vad_thread = VADThread()
    
    # Thread 3: TTS nhận câu trả lời để nói
    tts_thread = TTSThread()
    tts_thread.start()

    # Thread 2: STT + Chatbot. Truyền callback cho chatbot trả kết quả về TTS queue
    def on_final_result(text):
        print(f"[Main] STT final result: {text}")
        # Gửi câu trả lời Chatbot vào queue của TTS
        # Lấy phản hồi chatbot
        response = stt_thread.chatbot.get_response(text)
        print(f"[Main] Chatbot response: {response}")
        text_to_tts_queue.put(response)

    stt_thread = STTConversationThread(on_final_result=on_final_result)
    
    # Truyền queue cho TTS thread
    tts_thread.text_queue = text_to_tts_queue

    # 3. Start các thread
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

        wakeup_thread.stop()
        vad_thread.stop()
        stt_thread.stop()
        tts_thread.stop()

        wakeup_thread.join()
        vad_thread.join()
        stt_thread.join()
        tts_thread.join()

    print("System stopped cleanly.")


if __name__ == "__main__":
    main()
